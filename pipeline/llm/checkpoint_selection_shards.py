"""CPU-only integrity and exact aggregation for Step-05 validation shards."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def shard_indices(count, num_shards, shard_index):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    return list(range(shard_index, count, num_shards))


def metric_sums(predictions, top_k):
    ranks = [row["target_rank"] for row in predictions]
    hits = [rank for rank in ranks if rank is not None and rank <= top_k]
    return {
        "users": len(predictions), "hit_sum": len(hits), "recall_sum": len(hits),
        "ndcg_sum": math.fsum(1 / math.log2(rank + 1) for rank in hits),
        "mrr_sum": math.fsum(1 / rank for rank in hits),
        "short_ranking_users": sum(len(row["recommendations"]) < top_k for row in predictions),
    }


def metrics_from_predictions(predictions, top_k):
    sums = metric_sums(predictions, top_k)
    count = sums["users"]
    if not count:
        raise ValueError("Cannot evaluate an empty validation population")
    return {
        "users": count,
        **{f"{name}@{top_k}": sums[f"{name}_sum"] / count
           for name in ("hit", "recall", "ndcg", "mrr")},
        "short_ranking_users": sums["short_ranking_users"],
        "complete_ranking_rate": (count - sums["short_ranking_users"]) / count,
    }


def checkpoint_identity(path):
    path = Path(path)
    files = sorted(set(path.glob("adapter*")))
    files = [file for file in files if file.is_file()]
    if not files:
        raise ValueError(f"No adapter files at {path}")
    return {"path": str(path.resolve()),
            "files": {file.name: file_digest(file) for file in files}}


def run_identity(config, records, checkpoints, settings):
    ids = [str(row["user_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Validation population contains duplicate user IDs")
    return {
        "schema_version": 1, "model": config["model"]["name"],
        "config_sha256": digest(config), "validation_sha256": digest(records),
        "users": len(records), "settings": settings,
        "catalogue_sha256": file_digest(config["data"]["catalogue_file"]),
        "metadata_sha256": file_digest(config["data"]["metadata_file"]),
        "checkpoints": [checkpoint_identity(path) for path in checkpoints],
        "code_sha256": {name: file_digest(Path(__file__).with_name(name)) for name in (
            "05_select_best_checkpoint.py", "checkpoint_selection_shards.py",
            "common.py", "constrained_decoding.py")},
    }


def shard_path(directory, identity, checkpoint, num_shards, shard_index):
    model = re.sub(r"[^A-Za-z0-9_.-]", "_", identity["model"])
    return Path(directory) / (
        f"{model}__{Path(checkpoint).name}__{digest(str(checkpoint))[:12]}"
        f"__n{num_shards}__i{shard_index}.json")


def atomic_shard_write(path, payload):
    """Publish only completed JSON; refuse an existing shard, including races."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication without overwriting another job.
    finally:
        Path(temporary).unlink(missing_ok=True)


def make_payload(identity, checkpoint, num_shards, shard_index, indices, predictions, elapsed):
    if len(indices) != len(predictions):
        raise ValueError("Prediction/index count mismatch")
    return {
        "complete": True, "identity": identity, "checkpoint": str(checkpoint),
        "num_shards": num_shards, "shard_index": shard_index,
        "global_indices": indices, "predictions": predictions,
        "sums": metric_sums(predictions, identity["settings"]["top_k"]),
        "elapsed_seconds": elapsed,
    }


def aggregate_payloads(payloads, identity, checkpoint, num_shards, records):
    """Retain user contributions to avoid rounded averages or partition-dependent sums."""
    by_index = {}
    ordered = {}
    elapsed = 0.0
    for payload in payloads:
        if payload.get("complete") is not True:
            raise ValueError("Incomplete shard")
        if payload.get("checkpoint") != str(checkpoint):
            raise ValueError("Checkpoint mismatch")
        if payload.get("identity") != identity:
            raise ValueError("Shard metadata/validation/checkpoint identity mismatch")
        if payload.get("num_shards") != num_shards:
            raise ValueError("Shard count mismatch")
        index = payload["shard_index"]
        if index in by_index:
            raise ValueError(f"Duplicate shard {index}")
        expected = shard_indices(len(records), num_shards, index)
        if payload["global_indices"] != expected:
            raise ValueError("Shard membership mismatch")
        predictions = payload["predictions"]
        if len(predictions) != len(expected):
            raise ValueError("Shard user count mismatch")
        if payload["sums"] != metric_sums(predictions, identity["settings"]["top_k"]):
            raise ValueError("Shard metric sums mismatch")
        for global_index, prediction in zip(expected, predictions):
            record = records[global_index]
            if (prediction["user_id"] != str(record["user_id"])
                    or prediction["target_item_id"] != str(record["target_item_id"])):
                raise ValueError("Prediction/user/target mapping mismatch")
            ordered[global_index] = prediction
        by_index[index] = payload
        elapsed += payload["elapsed_seconds"]
    missing = set(range(num_shards)) - by_index.keys()
    if missing:
        raise ValueError(f"Missing shards: {sorted(missing)}")
    if len(ordered) != len(records) or len(records) != identity["users"]:
        raise ValueError("Incomplete validation population")
    predictions = [ordered[index] for index in range(len(records))]
    metrics = metrics_from_predictions(predictions, identity["settings"]["top_k"])
    metrics["elapsed_seconds"] = elapsed
    return metrics, predictions
