"""Prepare, rank, and promote recent5-three QLoRA tuning runs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "pipeline/configs/llm/tuning/video_games_recent5_three_sweep.yaml"
)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
    )


def write_yaml(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == text:
            return
        if not overwrite:
            raise FileExistsError(
                f"Generated config differs from existing file: {path}. "
                "Use --overwrite after reviewing the change."
            )
    atomic_write_text(path, text)


def manifest_values(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_yaml(path)
    workflow = manifest.get("workflow")
    models = manifest.get("models")
    candidates = manifest.get("stage1_candidates")
    if not isinstance(workflow, dict):
        raise ValueError("Manifest is missing workflow settings.")
    if not isinstance(models, dict) or set(models) != {"qwen", "llama"}:
        raise ValueError("Manifest models must contain qwen and llama.")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError("Stage 1 must define exactly three screening candidates.")

    allowed_learning_rates = {5e-5, 1e-4}
    allowed_ranks = {8, 16}
    allowed_dropouts = {0.05}
    allowed_epochs = {1}
    candidate_ids: set[str] = set()
    observed_learning_rates: set[float] = set()
    observed_ranks: set[int] = set()
    observed_dropouts: set[float] = set()
    observed_epochs: set[int] = set()
    observed_pairs: set[tuple[float, int]] = set()

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Every Stage-1 candidate must be a mapping.")
        candidate_id = str(candidate.get("id", ""))
        learning_rate = float(candidate["learning_rate"])
        rank = int(candidate["rank"])
        alpha = int(candidate["alpha"])
        dropout = float(candidate["dropout"])
        epochs = int(candidate["epochs"])
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError("Stage-1 candidate IDs must be unique and non-empty.")
        if learning_rate not in allowed_learning_rates:
            raise ValueError(f"Unsupported learning rate: {learning_rate}")
        if rank not in allowed_ranks or alpha != 2 * rank:
            raise ValueError("LoRA alpha must equal 2 x rank for rank 8/16.")
        if dropout not in allowed_dropouts:
            raise ValueError(f"Unsupported LoRA dropout: {dropout}")
        if epochs not in allowed_epochs:
            raise ValueError(f"Unsupported epoch count: {epochs}")
        candidate_ids.add(candidate_id)
        observed_learning_rates.add(learning_rate)
        observed_ranks.add(rank)
        observed_dropouts.add(dropout)
        observed_epochs.add(epochs)
        observed_pairs.add((learning_rate, rank))

    if (
        observed_learning_rates != allowed_learning_rates
        or observed_ranks != allowed_ranks
        or observed_dropouts != allowed_dropouts
        or observed_epochs != allowed_epochs
        or observed_pairs != {(5e-5, 8), (1e-4, 8), (1e-4, 16)}
    ):
        raise ValueError("Stage 1 must cover every requested tuning value.")
    return manifest, workflow


def validate_fixed_protocol(config: dict[str, Any], path: Path) -> None:
    expected = {
        "train_file": "train_recent5_three_per_user.jsonl",
        "validation_file": "validation_recent5.jsonl",
    }
    data = config.get("data", {})
    for key, basename in expected.items():
        if Path(str(data.get(key, ""))).name != basename:
            raise ValueError(f"{path}: data.{key} must use {basename}.")

    quantization = config.get("quantization", {})
    if quantization.get("compute_dtype") != "float16":
        raise ValueError(f"{path}: compute_dtype must be float16.")
    if quantization.get("quant_type") != "nf4":
        raise ValueError(f"{path}: quant_type must be nf4.")
    if quantization.get("use_double_quant") is not True:
        raise ValueError(f"{path}: double quantization must remain enabled.")

    training = config.get("training", {})
    if training.get("gradient_checkpointing") is not True:
        raise ValueError(f"{path}: gradient checkpointing must remain enabled.")
    if int(training.get("max_sequence_length", 0)) not in {512, 1024}:
        raise ValueError(f"{path}: max sequence length must be 512 or 1024.")

    for section in ("checkpoint_selection", "generation"):
        settings = config.get(section, {})
        if int(settings.get("max_history_length", 0)) != 5:
            raise ValueError(f"{path}: {section} history length must be 5.")
        if settings.get("exclude_history_items") is not True:
            raise ValueError(f"{path}: history exclusion must stay enabled.")


def relative_project_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def prepare_stage1(args: argparse.Namespace) -> None:
    manifest_path = project_path(args.manifest)
    manifest, workflow = manifest_values(manifest_path)
    generated_root = project_path(workflow["generated_config_dir"])
    output_root = str(workflow["output_root"]).rstrip("/")
    single_gpu = bool(args.single_gpu)
    rtx_3090 = bool(getattr(args, "rtx_3090", False))
    if rtx_3090:
        stage_name = "stage1_fast_3090"
    elif single_gpu:
        stage_name = "stage1_fast_single_gpu"
    else:
        stage_name = "stage1_fast"
    created: list[Path] = []

    for model_key, model_settings in manifest["models"].items():
        base_path = project_path(model_settings["base_config"])
        base = load_yaml(base_path)
        validate_fixed_protocol(base, base_path)

        for candidate in manifest["stage1_candidates"]:
            candidate_id = str(candidate["id"])
            config = copy.deepcopy(base)
            experiment_name = (
                f"{model_key}_video_games_recent5_three_"
                f"{stage_name}_{candidate_id}"
            )
            config["experiment"]["name"] = experiment_name
            config["experiment"]["seed"] = int(workflow["seed"])
            config["data"]["max_train_samples"] = int(
                workflow["stage1_train_samples"]
            )
            config["data"]["max_validation_samples"] = int(
                workflow["stage1_validation_loss_samples"]
            )
            config["data"]["shuffle_before_sampling"] = True
            config["lora"]["rank"] = int(candidate["rank"])
            config["lora"]["alpha"] = int(candidate["alpha"])
            config["lora"]["dropout"] = float(candidate["dropout"])
            config["training"]["learning_rate"] = float(
                candidate["learning_rate"]
            )
            config["training"]["epochs"] = int(candidate["epochs"])
            config["training"]["max_sequence_length"] = 512
            config["checkpoint_selection"]["num_beams"] = 10
            config["training"]["save_total_limit"] = int(
                candidate["epochs"]
            )
            if rtx_3090:
                config["training"]["train_batch_size"] = 2
                config["training"]["gradient_accumulation_steps"] = 32
            elif single_gpu:
                config["training"]["train_batch_size"] = 4
                config["training"]["gradient_accumulation_steps"] = 16
            config["training"]["output_dir"] = (
                f"{output_root}/{model_key}/{stage_name}/"
                f"{workflow['stage1_train_samples']}_samples/{candidate_id}"
            )
            config["checkpoint_selection"]["max_validation_users"] = int(
                workflow["stage1_validation_recommendation_users"]
            )
            config["checkpoint_selection"]["sampling_seed"] = int(
                workflow["validation_sampling_seed"]
            )
            config["checkpoint_selection"]["tie_breakers"] = [
                "hit",
                "mrr",
            ]
            config["tuning"] = {
                "workflow": workflow["name"],
                "stage": 1,
                "model_key": model_key,
                "candidate_id": candidate_id,
                "manifest": relative_project_path(manifest_path),
            }
            if rtx_3090:
                config["tuning"]["execution"] = "rtx_3090"
            elif single_gpu:
                config["tuning"]["execution"] = "single_gpu"
            validate_fixed_protocol(config, base_path)
            config_path = (
                generated_root
                / stage_name
                / f"{model_key}_{candidate_id}.yaml"
            )
            write_yaml(config_path, config, args.overwrite)
            created.append(config_path)

    list_path = generated_root / f"{stage_name}_configs.txt"
    atomic_write_text(
        list_path,
        "".join(f"{relative_project_path(path)}\n" for path in created),
    )
    if rtx_3090:
        label = "RTX-3090 Stage-1"
    elif single_gpu:
        label = "single-GPU Stage-1"
    else:
        label = "Stage-1"
    print(f"Prepared {len(created)} {label} configs.")
    print(f"Config list: {list_path}")


def summary_row(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_directory = Path(config["training"]["output_dir"])
    run_summary_path = output_directory / "run_summary.json"
    selection_path = (
        output_directory
        / "validation_selection"
        / "checkpoint_selection.json"
    )
    summary_path = output_directory / "experiment_summary.json"
    for required_path in (
        run_summary_path,
        selection_path,
        summary_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Incomplete run for {config_path}: missing {required_path}"
            )

    run_summary = read_json(run_summary_path)
    selection = read_json(selection_path)
    summary = read_json(summary_path)
    experiment_name = str(config["experiment"]["name"])
    if run_summary.get("experiment_name") != experiment_name:
        raise ValueError(
            f"Training summary does not match {config_path}."
        )
    if selection.get("experiment") != experiment_name:
        raise ValueError(
            f"Checkpoint selection does not match {config_path}."
        )
    if summary.get("experiment") != experiment_name:
        raise ValueError(
            f"Experiment summary does not match {config_path}."
        )

    selected_checkpoint = Path(str(summary.get("selected_checkpoint", "")))
    if not selected_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Selected checkpoint is missing: {selected_checkpoint}"
        )
    if selected_checkpoint.resolve() != Path(
        str(selection.get("best_checkpoint", ""))
    ).resolve():
        raise ValueError(
            f"Selected checkpoint disagrees with {selection_path}."
        )
    try:
        selected_checkpoint.resolve().relative_to(output_directory.resolve())
    except ValueError as error:
        raise ValueError(
            f"Selected checkpoint is outside its run directory: "
            f"{selected_checkpoint}"
        ) from error

    expected_values = {
        "model": config["model"]["name"],
        "seed": config["experiment"]["seed"],
        "learning_rate": config["training"]["learning_rate"],
        "lora_rank": config["lora"]["rank"],
        "lora_alpha": config["lora"]["alpha"],
        "lora_dropout": config["lora"]["dropout"],
        "epochs": config["training"]["epochs"],
    }
    for key, expected in expected_values.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{summary_path}: {key} does not match the config."
            )

    top_k = int(config["checkpoint_selection"].get("top_k", 10))
    best_metrics = selection.get("best_metrics", {})
    for metric_name in ("ndcg", "hit", "recall", "mrr"):
        selection_key = f"{metric_name}@{top_k}"
        summary_key = f"validation_{selection_key}"
        if summary.get(summary_key) != best_metrics.get(selection_key):
            raise ValueError(
                f"{summary_path}: {summary_key} does not match selection."
            )
    tuning = config.get("tuning", {})
    return {
        "model_key": tuning.get("model_key"),
        "stage": tuning.get("stage"),
        "candidate_id": tuning.get("candidate_id"),
        "experiment": summary["experiment"],
        "config_file": str(config_path),
        "model": summary["model"],
        "seed": summary["seed"],
        "learning_rate": summary["learning_rate"],
        "lora_rank": summary["lora_rank"],
        "lora_alpha": summary["lora_alpha"],
        "lora_dropout": summary["lora_dropout"],
        "epochs": summary["epochs"],
        "effective_global_batch_size": summary[
            "effective_global_batch_size"
        ],
        "warmup_steps": summary["warmup_steps"],
        "training_loss": summary["training_loss"],
        "validation_loss": summary["validation_loss"],
        f"validation_ndcg@{top_k}": summary[
            f"validation_ndcg@{top_k}"
        ],
        f"validation_hit@{top_k}": summary[
            f"validation_hit@{top_k}"
        ],
        f"validation_recall@{top_k}": summary[
            f"validation_recall@{top_k}"
        ],
        f"validation_mrr@{top_k}": summary[
            f"validation_mrr@{top_k}"
        ],
        "validation_users": summary["validation_users"],
        "validation_sampling_seed": summary["validation_sampling_seed"],
        "validation_sample_sha256": summary[
            "validation_sample_sha256"
        ],
        "full_validation_set_used": summary["full_validation_set_used"],
        "selected_checkpoint": summary["selected_checkpoint"],
        "runtime_seconds": summary["total_runtime_seconds"],
        "output_directory": str(output_directory),
    }


def listed_config_paths(
    config_directory: Path,
    config_list_argument: str | None,
) -> tuple[list[Path], Path]:
    if config_list_argument is not None:
        config_list_path = project_path(config_list_argument)
    else:
        list_name = f"{config_directory.name}_configs.txt"
        candidates = (
            config_directory / list_name,
            config_directory.parent / list_name,
        )
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise FileNotFoundError(
                "Could not identify one generated config list for "
                f"{config_directory}. Pass --config-list explicitly."
            )
        config_list_path = existing[0]

    if not config_list_path.is_file():
        raise FileNotFoundError(
            f"Config list does not exist: {config_list_path}"
        )

    config_paths = [
        project_path(line.strip())
        for line in config_list_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not config_paths:
        raise ValueError(f"Config list is empty: {config_list_path}")
    if len(config_paths) != len(set(config_paths)):
        raise ValueError(
            f"Config list contains duplicate paths: {config_list_path}"
        )

    listed_set = set(config_paths)
    discovered_set = {
        path.resolve() for path in config_directory.glob("*.yaml")
    }
    missing_configs = sorted(listed_set - discovered_set)
    unexpected_configs = sorted(discovered_set - listed_set)
    if missing_configs or unexpected_configs:
        details: list[str] = []
        if missing_configs:
            details.append(
                "missing listed configs: "
                + ", ".join(str(path) for path in missing_configs)
            )
        if unexpected_configs:
            details.append(
                "unlisted configs: "
                + ", ".join(str(path) for path in unexpected_configs)
            )
        raise ValueError(
            f"Config directory does not match {config_list_path}: "
            + "; ".join(details)
        )

    return config_paths, config_list_path


def rank_runs(args: argparse.Namespace) -> None:
    config_directory = project_path(args.config_dir)
    config_paths, config_list_path = listed_config_paths(
        config_directory=config_directory,
        config_list_argument=args.config_list,
    )
    if args.model_key:
        config_paths = [
            path for path in config_paths
            if load_yaml(path).get("tuning", {}).get("model_key") == args.model_key
        ]
        if not config_paths:
            raise ValueError(f"No configs found for {args.model_key}.")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for config_path in config_paths:
        try:
            rows.append(summary_row(config_path))
        except FileNotFoundError as error:
            missing.append(str(error))

    if missing and not args.allow_incomplete:
        raise FileNotFoundError(
            "Incomplete tuning runs:\n" + "\n".join(missing)
        )
    if not rows:
        raise ValueError("No completed runs are available to rank.")

    if args.require_full_validation and any(
        not row["full_validation_set_used"] for row in rows
    ):
        raise ValueError(
            "Full-validation ranking requested, but at least one run used "
            "a validation subset."
        )

    sample_keys = {
        (
            row["validation_sampling_seed"],
            row["validation_sample_sha256"],
            row["validation_users"],
        )
        for row in rows
    }
    if len(sample_keys) != 1:
        raise ValueError(
            "Candidate runs did not use the same validation-user sample."
        )

    top_k = 10
    ndcg_key = f"validation_ndcg@{top_k}"
    hit_key = f"validation_hit@{top_k}"
    mrr_key = f"validation_mrr@{top_k}"
    ranked: list[dict[str, Any]] = []
    best_by_model: dict[str, dict[str, Any]] = {}
    for model_key in sorted({str(row["model_key"]) for row in rows}):
        model_rows = [row for row in rows if row["model_key"] == model_key]
        model_rows.sort(
            key=lambda row: (
                -float(row[ndcg_key]),
                -float(row[hit_key]),
                -float(row[mrr_key]),
                float(row["validation_loss"]),
            )
        )
        for rank, row in enumerate(model_rows, start=1):
            row["rank_within_model"] = rank
            ranked.append(row)
        best_by_model[model_key] = model_rows[0]

    output_directory = project_path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "ranking_metric": ndcg_key,
        "tie_breakers": [hit_key, mrr_key, "validation_loss"],
        "config_list": str(config_list_path),
        "validation_sample": {
            "seed": next(iter(sample_keys))[0],
            "sha256": next(iter(sample_keys))[1],
            "users": next(iter(sample_keys))[2],
        },
        "incomplete_runs": missing,
        "best_by_model": best_by_model,
        "results": ranked,
    }
    ranking_path = output_directory / "tuning_results.json"
    write_json(ranking_path, payload)

    csv_path = output_directory / "tuning_results.csv"
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    os.replace(temporary, csv_path)
    winning_configs_path = output_directory / "winning_configs.txt"
    atomic_write_text(
        winning_configs_path,
        "".join(
            f"{best_by_model[model_key]['config_file']}\n"
            for model_key in ("qwen", "llama")
            if model_key in best_by_model
        ),
    )
    print(f"Ranked {len(ranked)} completed runs.")
    print(f"JSON: {ranking_path}")
    print(f"CSV:  {csv_path}")
    print(f"Winning configs: {winning_configs_path}")


def promote_stage2(args: argparse.Namespace) -> None:
    manifest_path = project_path(args.manifest)
    _, workflow = manifest_values(manifest_path)
    ranking = read_json(project_path(args.ranking))
    incomplete_runs = ranking.get("incomplete_runs", [])
    if incomplete_runs:
        raise ValueError(
            "Cannot promote from a ranking with incomplete runs."
        )
    results = ranking.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Ranking JSON has no results list.")

    top_per_model = int(workflow["promote_per_model"])
    if top_per_model != 1 or args.top_per_model != 1:
        raise ValueError("The fast protocol promotes exactly one candidate per model.")
    output_root = str(workflow["output_root"]).rstrip("/")
    config_directory = project_path(args.output_dir)
    promoted: list[Path] = []

    for model_key in ([args.model_key] if args.model_key else ("qwen", "llama")):
        winners = [
            row
            for row in results
            if row.get("model_key") == model_key
            and int(row.get("rank_within_model", 0)) <= top_per_model
        ]
        if len(winners) != top_per_model:
            raise ValueError(
                f"Expected {top_per_model} ranked {model_key} runs, "
                f"found {len(winners)}."
            )

        for winner in winners:
            config = load_yaml(project_path(winner["config_file"]))
            candidate_id = str(winner["candidate_id"])
            if config.get("tuning", {}).get("workflow") != workflow["name"]:
                raise ValueError("Cannot promote a candidate from the old protocol.")
            execution = config["tuning"].get("execution")
            stage_name = {
                "rtx_3090": "stage2_fast_3090",
                "single_gpu": "stage2_fast_single_gpu",
            }.get(execution, "stage2_fast")
            experiment_name = (
                f"{model_key}_video_games_recent5_three_{stage_name}_"
                f"{candidate_id}"
            )
            config["experiment"]["name"] = experiment_name
            config["data"]["max_train_samples"] = None
            config["data"]["max_validation_samples"] = int(
                workflow["stage2_validation_loss_samples"]
            )
            config["training"]["epochs"] = 1
            config["training"]["save_total_limit"] = 1
            config["checkpoint_selection"]["num_beams"] = 20
            config["training"]["output_dir"] = (
                f"{output_root}/{model_key}/{stage_name}/{candidate_id}"
            )
            config["checkpoint_selection"]["max_validation_users"] = None
            config["checkpoint_selection"]["sampling_seed"] = int(
                workflow["validation_sampling_seed"]
            )
            config["tuning"].update(
                {
                    "stage": 2,
                    "parent_stage1_experiment": winner["experiment"],
                    "parent_stage1_rank": winner["rank_within_model"],
                }
            )
            config_path = (
                config_directory / stage_name / f"{model_key}_{candidate_id}_full.yaml"
            )
            write_yaml(config_path, config, args.overwrite)
            promoted.append(config_path)

    for directory in sorted({path.parent for path in promoted}):
        suffix = f"_{args.model_key}" if args.model_key else ""
        list_path = directory / f"{directory.name}{suffix}_configs.txt"
        atomic_write_text(
            list_path,
            "".join(
                f"{relative_project_path(path)}\n"
                for path in promoted if path.parent == directory
            ),
        )
        print(f"Config list: {list_path}")
    print(f"Prepared {len(promoted)} Stage-2 configs.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-stage1")
    prepare.add_argument("--manifest", default=DEFAULT_MANIFEST)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(function=prepare_stage1, single_gpu=False)

    prepare_single_gpu = subparsers.add_parser(
        "prepare-stage1-single-gpu"
    )
    prepare_single_gpu.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
    )
    prepare_single_gpu.add_argument("--overwrite", action="store_true")
    prepare_single_gpu.set_defaults(
        function=prepare_stage1,
        single_gpu=True,
    )

    prepare_3090 = subparsers.add_parser("prepare-stage1-3090")
    prepare_3090.add_argument("--manifest", default=DEFAULT_MANIFEST)
    prepare_3090.add_argument("--overwrite", action="store_true")
    prepare_3090.set_defaults(
        function=prepare_stage1,
        single_gpu=False,
        rtx_3090=True,
    )

    rank = subparsers.add_parser("rank")
    rank.add_argument("--config-dir", required=True)
    rank.add_argument("--config-list")
    rank.add_argument("--output-dir", required=True)
    rank.add_argument("--allow-incomplete", action="store_true")
    rank.add_argument("--require-full-validation", action="store_true")
    rank.add_argument("--model-key", choices=("qwen", "llama"))
    rank.set_defaults(function=rank_runs)

    promote = subparsers.add_parser("promote-stage2")
    promote.add_argument("--manifest", default=DEFAULT_MANIFEST)
    promote.add_argument("--ranking", required=True)
    promote.add_argument(
        "--top-per-model",
        type=int,
        default=1,
        choices=(1,),
    )
    promote.add_argument(
        "--output-dir",
        default=(
            "pipeline/configs/llm/tuning/generated/fast_promoted"
        ),
    )
    promote.add_argument("--model-key", choices=("qwen", "llama"))
    promote.add_argument("--overwrite", action="store_true")
    promote.set_defaults(function=promote_stage2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
