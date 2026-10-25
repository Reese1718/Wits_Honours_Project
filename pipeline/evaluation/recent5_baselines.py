"""Context-window ablation orchestration; original training scripts are reused.

validate/status/export are CPU-only. audit prepares RecBole data, never a model.
Only run trains, using the existing Colab training entry points.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
COLAB_DATA_ROOT = Path("/content/drive/MyDrive/Wits_Honours_Project/datasets/")
MODELS = {"sasrec": "SASRec", "bert4rec": "BERT4Rec", "gru4rec": "GRU4Rec"}
DATASETS = {"fashion": ("Fashion", "fashion", 3135), "video_games": ("Video_Games", "games", 109258)}
PARENT_SHA256 = {
    "sasrec_fashion.yaml": "5ecc664753f2f5fde4cfa309e2706cb652ff3b8e6c845b1a307ea950b0518b5e",
    "bert4rec_fashion.yaml": "f7f3bfe14d0f934e810f6b19ad5636fd8c30d85ab2cddbb9e2e402bb25edfca0",
    "gru4rec_fashion.yaml": "3a691915c54a9f51d6c772f3cce5605099af76b44c790df6255a679b2c43a7ce",
    "sasrec_games.yaml": "4427ba0f646736b34cc9baa1eccce3c1285c590411a9cf93a7b0c9e37a086fc7",
    "bert4rec_games.yaml": "5ca876eb1f1716fa4e484712e3e171cf530f9b70f1fafdd4a269a61678e7d3ab",
    "gru4rec_games.yaml": "0b5e176853c57803e13d32ab0f7d9be22cdff736322cfab8ee6324f72b5b0757",
}
EXPORT_FILES = ("config.yaml", "parent_config.yaml", "runtime_config.yaml", "run_summary.json",
                "selected_checkpoint.json", "test_metrics.json", "per_user_metrics.jsonl",
                "data_audit.json", "complete.json")


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def configs(dataset, model):
    stem = f"{model}_{DATASETS[dataset][1]}"
    parent = ROOT / f"pipeline/configs/{stem}.yaml"
    recent = parent.with_name(stem + "_recent5.yaml")
    return parent, recent


def validate_config(dataset, model):
    parent, recent = configs(dataset, model)
    if hashlib.sha256(parent.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != PARENT_SHA256[parent.name]:
        raise ValueError(f"Frozen parent config changed: {parent}")
    before = parent.read_bytes()
    after = recent.read_bytes()
    if before.count(b"MAX_ITEM_LIST_LENGTH: 50") != 1 or after != before.replace(
            b"MAX_ITEM_LIST_LENGTH: 50", b"MAX_ITEM_LIST_LENGTH: 5"):
        raise ValueError(f"Recent5 config must differ only in MAX_ITEM_LIST_LENGTH: {recent}")
    original = yaml.safe_load(before)
    derived = yaml.safe_load(after)
    expected = dict(original, MAX_ITEM_LIST_LENGTH=5)
    if derived != expected or derived["seed"] != 42:
        raise ValueError("Scientific config or seed differs")
    if derived["eval_args"] != {"group_by": "user", "order": "TO", "split": {"LS": "valid_and_test"}, "mode": "full"}:
        raise ValueError("Unexpected baseline split/evaluation protocol")
    return parent, recent, derived


def resolve_inter(dataset, data_root, explicit=None):
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    base = Path(data_root)
    name = DATASETS[dataset][0]
    aliases = [name] + (["Video Games"] if dataset == "video_games" else [])
    paths = {path.resolve() for alias in aliases for filename in aliases
             for path in (base / alias / "processed" / f"{filename}.inter", base / alias / f"{filename}.inter")
             if path.is_file()}
    if not paths:
        raise FileNotFoundError(f"Original {name}.inter not found under {base}; use --inter-file")
    if len({sha256(path) for path in paths}) != 1:
        raise ValueError(f"Conflicting original interaction sources: {sorted(map(str, paths))}")
    return sorted(paths)[0]


def source_population(path):
    histories = defaultdict(list)
    items = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        fields = [field.split(":", 1)[0] for field in next(reader)]
        for position, values in enumerate(reader):
            if not values: continue
            record = dict(zip(fields, values))
            user, item = record["user_id"], record["item_id"]
            timestamp = float(record["timestamp"])
            if not user or not item or not math.isfinite(timestamp):
                raise ValueError("Invalid original interaction")
            histories[user].append((timestamp, position, item))
            items.add(item)
    for history in histories.values():
        history.sort(key=lambda event: (event[0], event[1]))
        if len(history) < 4:
            raise ValueError("Original population has fewer than four interactions per user")
    if not histories: raise ValueError("Empty original interaction source")
    return dict(histories), items


def evaluation_population(histories, validation, test, split_dir=None, groups_file=None):
    """Check native targets against source and optional authoritative LLM evidence."""
    users = sorted(histories)
    expected = {"validation": {user: history[-2][2] for user, history in histories.items()},
                "test": {user: history[-1][2] for user, history in histories.items()}}
    for name, observed in (("validation", validation), ("test", test)):
        if observed != expected[name]:
            bad = [user for user in set(observed) | set(expected[name]) if observed.get(user) != expected[name].get(user)]
            raise ValueError(f"Native {name} targets differ from original chronological source: {bad[:5]}")
    evidence = {}
    if split_dir is not None:
        split_dir = Path(split_dir)
        for name in ("validation", "test"):
            path = split_dir / f"{name}.jsonl"
            if path.is_file():
                rows = json_rows(path)
                mapping = {str(row["user_id"]): str(row["target_item_id"]) for row in rows}
                if len(rows) != len(mapping) or mapping != expected[name]:
                    raise ValueError(f"Authoritative {name} targets/users mismatch: {path}")
                evidence[name] = {"file": str(path), "sha256": sha256(path), "users": len(rows), "targets_equal": True}
                if name == "test":
                    users = [str(row["user_id"]) for row in rows]
                    for row in rows:
                        original_history = [event[2] for event in histories[str(row["user_id"])]][:-1]
                        if row["history_item_ids"] != original_history:
                            raise ValueError("Authoritative full test history differs from source")
    population = {
        user: {"record_index": index, "user_id": user, "target": expected["test"][user],
               "original_history_length": len(histories[user]),
               "group": "sparse" if len(histories[user]) <= 7 else "longer_history"}
        for index, user in enumerate(users)
    }
    if groups_file and Path(groups_file).is_file():
        with Path(groups_file).open(encoding="utf-8-sig", newline="") as stream:
            groups = list(csv.DictReader(stream))
        if len(groups) != len(population) or len({row["user_id"] for row in groups}) != len(groups):
            raise ValueError("Existing user_groups.csv population mismatch")
        for row in groups:
            expected_row = population.get(row["user_id"])
            if expected_row is None or any(str(expected_row[key]) != row[key] for key in
                    ("record_index", "user_id", "original_history_length", "group")):
                raise ValueError("Existing groups differ from ORIGINAL history lengths/index mapping")
        evidence["user_groups"] = {"file": str(groups_file), "sha256": sha256(groups_file), "aligned": True}
    return population, evidence


def metrics_at_10(rank):
    hit = float(rank is not None and 1 <= rank <= 10)
    return {"Hit@10": hit, "Recall@10": hit,
            "NDCG@10": 1 / math.log2(rank + 1) if hit else 0.0,
            "MRR@10": 1 / rank if hit else 0.0}


def validate_per_user(rows, population):
    seen = set()
    for row in rows:
        user = row["user_id"]
        if user not in population or user in seen:
            raise ValueError("Duplicate or unexpected evaluation user")
        reference = population[user]
        for key in ("record_index", "target", "original_history_length", "group"):
            if row[key] != reference[key]: raise ValueError(f"Per-user alignment mismatch: {key}")
        rank = row["target_rank"]
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 10):
            raise ValueError("Invalid target rank")
        expected = metrics_at_10(rank)
        if row["metrics"] != expected or any(not math.isfinite(value) or not 0 <= value <= 1 for value in row["metrics"].values()):
            raise ValueError("Invalid per-user metrics")
        seen.add(user)
    if seen != set(population): raise ValueError("Incomplete per-user evaluation population")
    return sorted(rows, key=lambda row: row["record_index"])


def validate_completion(directory):
    directory = Path(directory)
    complete = json.loads((directory / "complete.json").read_text())
    if complete.get("complete") is not True:
        raise ValueError("Run is incomplete")
    for name, fingerprint in complete["files"].items():
        if name not in EXPORT_FILES or name == "complete.json" or sha256(directory / name) != fingerprint:
            raise ValueError("Run export integrity mismatch")
    if set(complete["files"]) != set(EXPORT_FILES) - {"complete.json"}:
        raise ValueError("Incomplete export allowlist")
    return complete


def export_statistics(run_dir, destination):
    run_dir, destination = Path(run_dir).resolve(), Path(destination).resolve()
    validate_completion(run_dir)
    if destination.exists(): raise FileExistsError(f"Statistics export exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recent5-export-", dir=destination.parent) as directory:
        staged = Path(directory) / "bundle"
        staged.mkdir()
        for name in EXPORT_FILES: shutil.copy2(run_dir / name, staged / name)
        os.rename(staged, destination)
    return destination


def experiment_paths(dataset, model, output_root=None, statistics_root=None):
    run = Path(output_root or ROOT / "results/recent5_baselines") / dataset / model
    exported = Path(statistics_root or ROOT / "results/statistics") / dataset / "recent5_baselines" / model
    return run.resolve(), exported.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "audit", "run", "status", "export"))
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=COLAB_DATA_ROOT)
    parser.add_argument("--inter-file", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--groups-file", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--statistics-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    models = list(MODELS) if args.model == "all" else [args.model]
    for dataset in datasets:
        for model in models:
            parent, recent, _ = validate_config(dataset, model)
            print(f"Verified {recent.name}: sole YAML difference MAX_ITEM_LIST_LENGTH 50 -> 5", flush=True)
    if args.command == "validate": return
    if args.command == "status":
        incomplete = []
        for dataset in datasets:
            for model in models:
                run, exported = experiment_paths(dataset, model, args.output_root, args.statistics_root)
                try:
                    for directory in (run, exported):
                        validate_completion(directory)
                    print(f"COMPLETE {dataset}/{model}: {exported}")
                except (OSError, ValueError, KeyError) as error:
                    incomplete.append(f"{dataset}/{model}: {error}")
        if incomplete: raise SystemExit("Incomplete experiments:\n" + "\n".join(incomplete))
        return
    if len(datasets) != 1 or len(models) != 1:
        parser.error("audit/run/export require one dataset and one model")
    dataset, model = datasets[0], models[0]
    run, exported = experiment_paths(dataset, model, args.output_root, args.statistics_root)
    if args.command == "export":
        print(export_statistics(run, exported)); return
    source = resolve_inter(dataset, args.data_root, args.inter_file)
    split_dir = args.split_dir or source.parent / "llm/splits"
    groups = args.groups_file or Path(args.statistics_root or ROOT / "results/statistics") / dataset / "user_groups.csv"
    from recent5_recbole import audit_data, train_existing
    if args.command == "audit":
        report, _ = audit_data(dataset, model, source, split_dir, groups)
        if args.audit_output: atomic_json(args.audit_output, report)
        print(json.dumps(report, indent=2)); return
    train_existing(dataset, model, source, split_dir, groups, run, exported)


if __name__ == "__main__":
    main()
