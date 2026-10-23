"""Export compact, validated Fashion QLoRA evidence for paired statistics.

Only explicit summary/per-user allowlists are copied. No checkpoint directory,
weights, raw validation predictions, or final test prediction dump is exported.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import yaml
from common import resolve_project_path

REQUIRED_METRICS = {f"{metric}@{k}" for k in (1, 5, 10) for metric in ("Hit", "Recall", "NDCG", "MRR")}
ROOT_SUMMARIES = ("run_summary.json", "train_results.json", "validation_results.json", "all_results.json")


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def user_groups(records):
    groups = []
    ids = set()
    for index, record in enumerate(records):
        user_id = str(record["user_id"])
        if not user_id or user_id in ids or not record["history_item_ids"]:
            raise ValueError("Test population has empty/duplicate users or empty histories")
        ids.add(user_id)
        length = len(record["history_item_ids"]) + 1
        groups.append(dict(record_index=index, user_id=user_id, original_history_length=length,
                           group="sparse" if length <= 7 else "longer_history"))
    if not groups:
        raise ValueError("Empty test population")
    return groups


def validate_per_user(rows, records):
    expected = {index: record for index, record in enumerate(records)}
    seen = set()
    for row in rows:
        index = row["record_index"]
        if index in seen or index not in expected:
            raise ValueError("Duplicate or unexpected per-user record_index")
        record = expected[index]
        if row["user_id"] != str(record["user_id"]) or row["target"] != str(record["target_item_id"]):
            raise ValueError("Per-user metrics do not align with authoritative test users/targets")
        values = row["metrics"]
        if not REQUIRED_METRICS.issubset(values) or any(
                not isinstance(values[name], (int, float)) or not math.isfinite(values[name])
                for name in REQUIRED_METRICS):
            raise ValueError("Missing or invalid per-user metrics at K=1,5,10")
        seen.add(index)
    if seen != set(expected):
        raise ValueError("Incomplete per-user metric population")


def model_bundle(config_path):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (config["experiment"]["dataset"] != "Fashion"
            or config.get("tuning", {}).get("stage") != 2
            or config["tuning"].get("workflow") != "fashion_recent5_three_fast"):
        raise ValueError("Statistics export requires completed Fashion Stage-2 configs")
    model = config["tuning"]["model_key"]
    if model not in ("qwen", "llama"):
        raise ValueError("Unknown Fashion model")
    root = resolve_project_path(config["training"]["output_dir"])
    test_records = json_rows(resolve_project_path(config["data"]["test_split_file"]))
    validation_records = json_rows(resolve_project_path(config["data"]["validation_split_file"]))
    groups = user_groups(test_records)
    prediction_path = resolve_project_path(config.get("generation", {}).get(
        "output_file", root / "test_predictions/test_predictions.jsonl"))
    metrics_path = resolve_project_path(config.get("evaluation", {}).get(
        "metrics_file", root / "test_predictions/test_metrics.json"))
    per_user_path = resolve_project_path(config.get("evaluation", {}).get(
        "per_user_metrics_file", root / "test_predictions/per_user_metrics.jsonl"))
    selection_path = resolve_project_path(config.get("generation", {}).get(
        "checkpoint_selection_file", root / "validation_selection/checkpoint_selection.json"))
    generation_path = prediction_path.parent / "generation_summary.json"
    metrics = json_file(metrics_path)
    generation = json_file(generation_path)
    selection = json_file(selection_path)
    run = json_file(root / "run_summary.json")
    validate_per_user(json_rows(per_user_path), test_records)
    count = len(groups)
    if (metrics["dataset"] != "Fashion" or metrics["model"] != config["model"]["name"]
            or metrics["counts"]["evaluated_users"] != count
            or metrics["counts"]["test_records"] != count
            or metrics["counts"]["prediction_records"] != count
            or metrics["counts"]["missing_prediction_records"] != 0
            or metrics.get("diagnostics", {}).get("incomplete_debug_evaluation", False)):
        raise ValueError("Test metrics are incomplete or belong to another experiment")
    if not REQUIRED_METRICS.issubset(metrics["metrics"]) or not {
            f"CatalogueCoverage@{k}" for k in (1, 5, 10)}.issubset(metrics["coverage"]):
        raise ValueError("Incomplete aggregate metric/cutoff schema")
    if (generation["completed_record_count"] != count or generation["test_record_count"] != count
            or generation["base_model"] != config["model"]["name"]
            or generation["num_beams"] != 10 or generation.get("debug_max_users") is not None):
        raise ValueError("Generation summary is incomplete or incompatible")
    if (selection["validation_users"] != len(validation_records)
            or selection.get("full_validation_set_used") is not True
            or selection["base_model"] != config["model"]["name"]
            or selection["experiment"] != config["experiment"]["name"]):
        raise ValueError("Checkpoint selection did not use this full Fashion validation set")
    selected = Path(selection["best_checkpoint"]).resolve()
    if selected != Path(generation["selected_checkpoint"]).resolve() or selected != Path(metrics["selected_checkpoint"]).resolve():
        raise ValueError("Selected checkpoint differs across final artifacts")
    if (run["dataset"] != "Fashion" or run["base_model"] != config["model"]["name"]
            or run["experiment_name"] != config["experiment"]["name"]
            or run["training_examples"] != len(json_rows(resolve_project_path(config["data"]["train_file"])))):
        raise ValueError("Training summary does not match the full Fashion training population")
    copies = [(root / name, Path(name)) for name in ROOT_SUMMARIES]
    copies += [(per_user_path, Path("per_user_metrics.jsonl")), (metrics_path, Path("test_metrics.json")),
               (generation_path, Path("generation_summary.json")),
               (selection_path, Path("validation_selection/checkpoint_selection.json")),
               (config_path, Path("config.yaml"))]
    checkpoints = selection.get("all_checkpoints", [])
    if not checkpoints:
        raise ValueError("Missing per-checkpoint validation summaries")
    for checkpoint in checkpoints:
        source = resolve_project_path(checkpoint["metrics_file"])
        source.relative_to(root.resolve())
        if source.parent != selection_path.parent or source.name != f"{checkpoint['checkpoint_name']}_metrics.json":
            raise ValueError("Unexpected per-checkpoint metric summary path")
        if json_file(source)["users"] != len(validation_records):
            raise ValueError("Incomplete per-checkpoint validation population")
        copies.append((source, Path("validation_selection") / source.name))
    for source, _ in copies:
        if not source.is_file():
            raise FileNotFoundError(source)
    return model, test_records, groups, copies


def export_bundle(config_paths, output):
    bundles = [model_bundle(path) for path in config_paths]
    if len(bundles) != 2 or {bundle[0] for bundle in bundles} != {"qwen", "llama"}:
        raise ValueError("Exactly one Qwen and one Llama Fashion result are required")
    if bundles[0][1] != bundles[1][1]:
        raise ValueError("Qwen and Llama authoritative test populations differ")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Statistics bundle already exists: {output}; choose a new export directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fashion-export-", dir=output.parent) as directory:
        staged = Path(directory) / "bundle"
        staged.mkdir()
        with (staged / "user_groups.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("record_index", "user_id", "original_history_length", "group"))
            writer.writeheader()
            writer.writerows(bundles[0][2])
        for model, _, _, copies in bundles:
            for source, relative in copies:
                destination = staged / f"{model}_qlora" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        os.rename(staged, output)
    print(f"Exported Fashion statistics bundle: {output}; users={len(bundles[0][2])}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/statistics/fashion"))
    args = parser.parse_args()
    export_bundle(args.configs, args.output_dir)


if __name__ == "__main__":
    main()
