"""Fashion recent5-three preparation and dependency-based cluster orchestration.

Submission commands print a plan by default. Only --submit invokes sbatch.
Training, selection and inference use the existing generic pipeline unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "pipeline/configs/llm/tuning/fashion_recent5_three_sweep.yaml"
GENERATED = ROOT / "pipeline/configs/llm/tuning/generated/fashion"
RANKING = ROOT / "pipeline/configs/llm/tuning/results/fashion_stage1_fast_3090"
PROMOTED = GENERATED / "fast_promoted/stage2_fast_3090"
GPU = ROOT / "slurm/llm/fashion_qlora_gpu.slurm"
CPU = ROOT / "slurm/llm/fashion_qlora_cpu.slurm"


def run_step(name, *arguments):
    subprocess.run([sys.executable, str(ROOT / "pipeline/llm" / name), *map(str, arguments)],
                   check=True, cwd=ROOT)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_data(data_root):
    data_root = Path(data_root).resolve()
    inter = data_root / "processed/Fashion.inter"
    metadata = data_root / "exports/Fashion_metadata_lookup.json"
    splits = data_root / "processed/llm/splits"
    sft = data_root / "processed/llm/sft"
    for path in (inter, metadata):
        if not path.is_file():
            raise FileNotFoundError(path)
    names = ("train_sequences.jsonl", "validation.jsonl", "test.jsonl", "item_catalogue.json")
    existing = [(splits / name).exists() for name in names]
    if any(existing) and not all(existing):
        raise ValueError("Incomplete existing split directory; review it before preparation")
    if all(existing):
        # Verify the existing authoritative splits against the original .inter;
        # never replace historical targets or the old split summary.
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / "splits"
            run_step("01_create_temporal_splits.py", "--dataset", "Fashion", "--inter-file", inter,
                     "--output-dir", fresh)
            for name in names:
                expected = read_json(fresh / name) if name == "item_catalogue.json" else read_rows(fresh / name)
                observed = read_json(splits / name) if name == "item_catalogue.json" else read_rows(splits / name)
                if expected != observed:
                    raise ValueError(f"Existing authoritative split differs from Fashion.inter: {name}")
    else:
        run_step("01_create_temporal_splits.py", "--dataset", "Fashion", "--inter-file", inter,
                 "--output-dir", splits)
    train_name = "train_recent5_three_per_user.jsonl"
    validation_name = "validation_recent5.jsonl"
    summary_name = "sft_summary_recent5_three.json"
    report_name = "validation_report_recent5_three.json"
    present = [(sft / name).exists() for name in (train_name, validation_name, summary_name)]
    if any(present) and not all(present):
        raise ValueError("Incomplete recent5-three SFT outputs; review before regenerating")
    if not any(present):
        run_step("02_build_sft_datasets.py", "--dataset", "Fashion", "--split-dir", splits,
                 "--metadata-file", metadata, "--output-dir", sft, "--max-history-length", 5,
                 "--max-training-examples-per-user", 3, "--train-output-name", train_name,
                 "--validation-output-name", validation_name, "--summary-output-name", summary_name,
                 "--allow-missing-metadata")
    summary = read_json(sft / summary_name)
    if (summary["maximum_history_length"] != 5
            or summary["maximum_training_examples_per_user"] != 3):
        raise ValueError("Existing SFT outputs do not implement recent5-three")
    # The existing Fashion metadata has one untitled catalogue item. The shared
    # canonical builder's 'Unknown title' fallback is retained and audited.
    run_step("03_validate_sft_datasets.py", "--dataset", "Fashion", "--split-dir", splits,
             "--sft-dir", sft, "--metadata-file", metadata, "--train-sft-name", train_name,
             "--validation-sft-name", validation_name, "--sft-summary-name", summary_name,
             "--report-name", report_name, "--allow-missing-metadata")
    counts = {
        "train": len(read_rows(sft / train_name)), "validation": len(read_rows(splits / "validation.jsonl")),
        "test": len(read_rows(splits / "test.jsonl")),
    }
    if summary["training_examples"] != counts["train"] or summary["validation_examples"] != counts["validation"]:
        raise ValueError("SFT report and actual record counts disagree")
    if counts["validation"] != counts["test"]:
        raise ValueError("Fashion validation/test populations differ in size")
    print("Validated actual Fashion counts:", json.dumps(counts), flush=True)
    return counts


def fashion_configs(paths, stage):
    configs = []
    for path in paths:
        path = Path(path).resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (config["experiment"]["dataset"] != "Fashion"
                or config.get("tuning", {}).get("workflow") != "fashion_recent5_three_fast"
                or config["tuning"]["stage"] != stage):
            raise ValueError(f"Wrong Fashion workflow/stage: {path}")
        for section in ("checkpoint_selection", "generation"):
            if (config[section]["num_beams"] != 10 or config[section]["batch_size"] != 4
                    or config[section]["max_history_length"] != 5):
                raise ValueError(f"Unexpected Fashion decoding settings: {path}")
        configs.append((path, config))
    return configs


def promoted_configs():
    paths = sorted(PROMOTED.glob("*_full.yaml"))
    configs = fashion_configs(paths, 2)
    if len(configs) != 2 or {cfg["tuning"]["model_key"] for _, cfg in configs} != {"qwen", "llama"}:
        raise ValueError("Expected exactly one promoted Fashion config per model")
    return [path for path, _ in configs]


def check_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    stage = config.get("tuning", {}).get("stage")
    if stage not in (1, 2):
        raise ValueError("Expected Stage 1 or Stage 2")
    fashion_configs([path], stage)
    training = config["training"]
    quantization = config["quantization"]
    if (training["train_batch_size"] != 2 or training["gradient_accumulation_steps"] != 32
            or training["max_sequence_length"] != 512 or training["epochs"] != 1
            or quantization["compute_dtype"] != "float16" or quantization["quant_type"] != "nf4"
            or quantization["use_double_quant"] is not True
            or config["data"]["max_train_samples"] is not None):
        raise ValueError("Fashion must retain full-data FP16/NF4 training and effective batch 64")
    # Tuning checks validation only; held-out test predictions/metrics are never
    # read to choose candidates or checkpoints.
    for split in ("validation",):
        records = read_rows(config["data"][f"{split}_split_file"])
        if not records or len({str(record["user_id"]) for record in records}) != len(records):
            raise ValueError(f"Empty or duplicate {split} population")
    expected_cap = 1000 if stage == 1 else None
    if config["checkpoint_selection"]["max_validation_users"] != expected_cap:
        raise ValueError("Incorrect Fashion validation population cap")
    if config["data"]["max_validation_samples"] != expected_cap:
        raise ValueError("Incorrect Fashion validation-loss cap")
    if not read_rows(config["data"]["train_file"]):
        raise ValueError("Empty Fashion training file")
    print(f"Fashion Stage-{stage} config and actual data populations verified: {path}")


def stage1_plan(paths):
    configs = fashion_configs(paths, 1)
    pairs = {(cfg["tuning"]["model_key"], cfg["tuning"]["candidate_id"]) for _, cfg in configs}
    if (len(configs) != 6 or len(pairs) != 6
            or any(sum(model == key for model, _ in pairs) != 3 for key in ("qwen", "llama"))):
        raise ValueError("Expected exactly three unique Fashion Stage-1 candidates per model")
    return [(f"stage1_{i}", [], GPU, ["stage1", str(path)]) for i, (path, _) in enumerate(configs)] + [
        ("rank", [f"stage1_{i}" for i in range(6)], CPU, ["rank"])]


def final_plan(paths):
    configs = fashion_configs(paths, 2)
    if len(configs) != 2 or {cfg["tuning"]["model_key"] for _, cfg in configs} != {"qwen", "llama"}:
        raise ValueError("Expected exactly one Stage-2 config per Fashion model")
    plan = []
    for path, config in configs:
        model = config["tuning"]["model_key"]
        previous = None
        for action, launcher in (("train", GPU), ("select", GPU), ("generate", GPU), ("evaluate", CPU)):
            label = f"{model}_{action}"
            plan.append((label, [previous] if previous else [], launcher, [action, str(path)]))
            previous = label
    plan.append(("compare_export", ["qwen_evaluate", "llama_evaluate"], CPU,
                 ["compare-export", *map(str, paths)]))
    return plan


def submit_plan(plan, submit=False):
    job_ids = {}
    for label, dependencies, launcher, arguments in plan:
        command = ["sbatch", "--parsable", "--job-name", f"fashion-{label}"]
        if dependencies:
            command += ["--dependency", "afterok:" + ":".join(job_ids[dep] for dep in dependencies)]
        command += [str(launcher), *arguments]
        print(shlex.join(command), flush=True)
        if submit:
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"Submission failed for {label}; already submitted jobs: {job_ids}. "
                    f"Inspect those jobs before retrying. Slurm: {error.stderr}"
                ) from error
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError(f"Invalid sbatch response: {result.stdout!r}")
            print(f"{label}: {job_id}", flush=True)
            job_ids[label] = job_id
        else:
            job_ids[label] = "{" + label + "}"
    return job_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, default=Path("/datasets/dnaidoo3/Fashion"))
    commands.add_parser("prepare-configs")
    commands.add_parser("rank")
    commands.add_parser("promote")
    check = commands.add_parser("check-config")
    check.add_argument("config", type=Path)
    for name in ("submit-stage1", "submit-final"):
        sub = commands.add_parser(name)
        sub.add_argument("--submit", action="store_true", help="Actually invoke sbatch; otherwise print plan")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_data(args.data_root)
        run_step("09_tuning_workflow.py", "prepare-stage1-3090", "--manifest", MANIFEST)
    elif args.command == "prepare-configs":
        run_step("09_tuning_workflow.py", "prepare-stage1-3090", "--manifest", MANIFEST)
    elif args.command == "rank":
        run_step("09_tuning_workflow.py", "rank", "--config-dir", GENERATED / "stage1_fast_3090",
                 "--config-list", GENERATED / "stage1_fast_3090_configs.txt", "--output-dir", RANKING)
    elif args.command == "promote":
        run_step("09_tuning_workflow.py", "promote-stage2", "--manifest", MANIFEST,
                 "--ranking", RANKING / "tuning_results.json", "--output-dir", GENERATED / "fast_promoted")
    elif args.command == "submit-stage1":
        paths = [ROOT / line for line in (GENERATED / "stage1_fast_3090_configs.txt").read_text().splitlines() if line]
        submit_plan(stage1_plan(paths), args.submit)
    elif args.command == "submit-final":
        submit_plan(final_plan(promoted_configs()), args.submit)
    elif args.command == "check-config":
        check_config(args.config)


if __name__ == "__main__":
    main()
