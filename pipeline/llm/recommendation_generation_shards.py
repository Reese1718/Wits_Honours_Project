"""CPU-only Step-06 shard publication and lossless prediction merging.

Targets participate only in identity/output validation, never prompt construction
or ranking. No recommendation metrics are calculated here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
import time

from checkpoint_selection_shards import (
    atomic_shard_write, checkpoint_identity, digest, file_digest, shard_indices,
)


def generation_identity(config, model, checkpoint, selection_file, test_file,
                        metadata_file, catalogue_file, population, settings):
    ids = [row[0] for row in population]
    if not population or any(not user_id for user_id in ids):
        raise ValueError("Test population is empty or contains empty user IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Test population contains duplicate user IDs")
    if settings["num_beams"] != 10:
        raise ValueError("Sharded Step 06 requires num_beams=10")
    return {
        "schema_version": 1, "model": model,
        "selected_checkpoint": checkpoint_identity(checkpoint),
        "selection_sha256": file_digest(selection_file),
        "config_sha256": digest(config), "test_sha256": file_digest(test_file),
        "population_sha256": digest(population), "users": len(population),
        "metadata_sha256": file_digest(metadata_file),
        "catalogue_sha256": file_digest(catalogue_file), "generation": settings,
        "code_sha256": {name: file_digest(Path(__file__).with_name(name)) for name in (
            "06_generate_test_recommendations.py", "recommendation_generation_shards.py",
            "common.py", "checkpoint_selection_shards.py")},
    }


def generation_shard_path(directory, identity, num_shards, shard_index):
    model = re.sub(r"[^A-Za-z0-9_.-]", "_", identity["model"])
    checkpoint = Path(identity["selected_checkpoint"]["path"]).name
    return Path(directory) / f"{model}__{checkpoint}__n{num_shards}__i{shard_index}.json"


def validate_rows(predictions, indices, population, top_k):
    if len(predictions) != len(indices):
        raise ValueError("Shard prediction count mismatch")
    for index, prediction in zip(indices, predictions):
        user_id, history, target = population[index]
        if (prediction.get("record_index") != index or prediction.get("user_id") != user_id
                or prediction.get("history") != history or prediction.get("target") != target):
            raise ValueError("Prediction index/user/history/target mapping mismatch")
        recommendations = prediction.get("recommendations")
        scores = prediction.get("scores")
        if (not isinstance(recommendations, list) or not isinstance(scores, list)
                or len(recommendations) != len(scores)
                or len(recommendations) > top_k
                or len(set(recommendations)) != len(recommendations)
                or prediction.get("requested_top_k") != top_k
                or prediction.get("generated_recommendation_count") != len(recommendations)):
            raise ValueError("Invalid Step-06 prediction schema")


def generate_shard(destination, identity, records, population, num_shards, shard_index,
                   batch_size, generate):
    if Path(destination).exists():
        raise FileExistsError(f"Completed shard already exists: {destination}; use a fresh run ID")
    indices = shard_indices(len(records), num_shards, shard_index)
    if not indices:
        raise ValueError("Empty shard; reduce num_shards")
    started = time.monotonic()
    predictions = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        results = generate([(index, records[index]) for index in batch_indices])
        validate_rows(results, batch_indices, population, identity["generation"]["top_k"])
        predictions.extend(results)
        completed = len(predictions)
        if completed % 400 < len(batch_indices) or completed == len(indices):
            elapsed = time.monotonic() - started
            rate = completed / max(elapsed, 1e-9)
            print(f"Generated {completed:,}/{len(indices):,} | Elapsed {elapsed:.0f}s | "
                  f"Users/sec {rate:.2f} | ETA {(len(indices)-completed)/rate:.0f}s", flush=True)
    payload = {
        "complete": True, "identity": identity, "num_shards": num_shards,
        "shard_index": shard_index, "users": len(indices), "global_indices": indices,
        "predictions": predictions, "elapsed_seconds": time.monotonic() - started,
    }
    atomic_shard_write(destination, payload)
    print(f"Completed shard: {destination}", flush=True)
    return payload


def merge_payloads(payloads, identity, population, num_shards):
    if identity["generation"]["num_beams"] != 10:
        raise ValueError("Merge requires num_beams=10")
    seen = set()
    ordered = {}
    for payload in payloads:
        if payload.get("complete") is not True:
            raise ValueError("Incomplete shard")
        if payload.get("identity") != identity:
            raise ValueError("Shard model/checkpoint/beam/test/config/generation identity mismatch")
        if payload.get("num_shards") != num_shards:
            raise ValueError("Shard count mismatch")
        index = payload["shard_index"]
        if index in seen:
            raise ValueError(f"Duplicate shard index: {index}")
        expected = shard_indices(len(population), num_shards, index)
        if payload.get("global_indices") != expected or payload.get("users") != len(expected):
            raise ValueError("Shard membership/user-count mismatch")
        validate_rows(payload["predictions"], expected, population, identity["generation"]["top_k"])
        ordered.update(zip(expected, payload["predictions"]))
        seen.add(index)
    missing = set(range(num_shards)) - seen
    if missing:
        raise ValueError(f"Missing shards: {sorted(missing)}")
    if len(ordered) != len(population) or len(population) != identity["users"]:
        raise ValueError("Incomplete test-user coverage")
    return [ordered[index] for index in range(len(population))]


def load_and_merge(directory, identity, population, num_shards):
    expected = {generation_shard_path(directory, identity, num_shards, index)
                for index in range(num_shards)}
    actual = set(Path(directory).glob("*.json"))
    if expected != actual:
        raise ValueError(f"Missing shards: {sorted(map(str, expected - actual))}; "
                         f"unexpected/duplicate shards: {sorted(map(str, actual - expected))}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(expected)]
    return merge_payloads(payloads, identity, population, num_shards), payloads


def publish_merged(predictions_file, predictions, summary, *, overwrite=False):
    """Publish a fully validated JSONL atomically; never append/resume a final merge."""
    predictions_file = Path(predictions_file)
    summary_file = predictions_file.parent / "generation_summary.json"
    for path in (predictions_file, summary_file):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Existing final output: {path}; explicit --overwrite required")
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".merge-partial-", dir=predictions_file.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, predictions_file)
        else:
            os.link(temporary, predictions_file)
    finally:
        Path(temporary).unlink(missing_ok=True)
    if overwrite:
        descriptor, temporary = tempfile.mkstemp(prefix=".summary-partial-", dir=summary_file.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(summary, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, summary_file)
        finally:
            Path(temporary).unlink(missing_ok=True)
    else:
        atomic_shard_write(summary_file, summary)
