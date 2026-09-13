"""
Step 08: Compare final LLM recommender experiments.

This script:

1. Loads Step 07 metrics from multiple experiment configurations.
2. Groups experiments by dataset.
3. Ranks models within each dataset.
4. Creates paired model comparisons using per-user results.
5. Optionally includes RecBole baseline results.
6. Saves:
      - comparison_results.json
      - comparison_results.csv
      - comparison_table.md

Datasets are never ranked against one another. Models are compared only when
they were evaluated on the same dataset and test users.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def nested_value(
    config: dict[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    value: Any = config

    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default

        value = value[key]

    return value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML file: {path}")

    return config


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")

    os.replace(temporary_path, path)


def find_repository_root(config_path: Path) -> Path:
    candidates = [
        Path.cwd().resolve(),
        config_path.resolve().parent,
        *config_path.resolve().parents,
    ]

    for candidate in candidates:
        if (candidate / "pipeline" / "llm").is_dir():
            return candidate

    return Path.cwd().resolve()


def resolve_path(
    value: str | Path,
    repository_root: Path,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (repository_root / path).resolve()


def dataset_name_from_config(config: dict[str, Any]) -> str:
    for key in ("experiment.dataset", "dataset.name"):
        value = nested_value(config, key)
        if value is not None and value != "":
            return str(value)

    dataset_value = config.get("dataset")
    if isinstance(dataset_value, (str, int, float, bool)):
        return str(dataset_value)

    if isinstance(dataset_value, dict) and dataset_value.get("dataset_name"):
        return str(dataset_value["dataset_name"])

    return "Unknown dataset"


def model_family(model_name: str) -> str:
    lowered = model_name.lower()

    if "qwen" in lowered:
        return "Qwen"

    if "llama" in lowered:
        return "Llama"

    return model_name.split("/")[-1]


def confidence_interval_95(
    values: list[float],
) -> tuple[float, float, float]:
    """
    Return the mean and normal-approximation 95% confidence interval.
    """

    if not values:
        return 0.0, 0.0, 0.0

    count = len(values)
    mean = sum(values) / count

    if count <= 1:
        return mean, mean, mean

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / (count - 1)

    standard_error = math.sqrt(variance / count)
    margin = 1.96 * standard_error

    return (
        mean,
        mean - margin,
        mean + margin,
    )


# ---------------------------------------------------------------------------
# Experiment loading
# ---------------------------------------------------------------------------

def load_experiment(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    repository_root = find_repository_root(config_path)

    dataset_name = dataset_name_from_config(config)

    model_name = str(
        nested_value(
            config,
            "model.name_or_path",
            default="Unknown model",
        )
    )

    display_name = nested_value(
        config,
        "experiment.display_name",
        default=None,
    )

    if not display_name:
        display_name = (
            f"{model_family(model_name)} QLoRA"
        )

    output_directory_value = nested_value(
        config,
        "training.output_dir",
    )

    if not output_directory_value:
        raise ValueError(
            f"{config_path} does not contain training.output_dir."
        )

    output_directory = resolve_path(
        output_directory_value,
        repository_root,
    )

    metrics_file_value = nested_value(
        config,
        "evaluation.metrics_file",
        default=(
            output_directory
            / "test_predictions"
            / "test_metrics.json"
        ),
    )

    per_user_file_value = nested_value(
        config,
        "evaluation.per_user_metrics_file",
        default=(
            output_directory
            / "test_predictions"
            / "per_user_metrics.jsonl"
        ),
    )

    metrics_file = resolve_path(
        metrics_file_value,
        repository_root,
    )

    per_user_file = resolve_path(
        per_user_file_value,
        repository_root,
    )

    if not metrics_file.is_file():
        raise FileNotFoundError(
            f"Step 07 metrics were not found for "
            f"{config_path.name}:\n  {metrics_file}"
        )

    results = read_json(metrics_file)

    if not isinstance(results, dict):
        raise ValueError(
            f"Invalid Step 07 metrics file: {metrics_file}"
        )

    metrics = results.get("metrics", {})
    coverage = results.get("coverage", {})

    if not isinstance(metrics, dict):
        raise ValueError(
            f"The metrics field is invalid in {metrics_file}"
        )

    if not isinstance(coverage, dict):
        coverage = {}

    combined_metrics = {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }

    combined_metrics.update(
        {
            str(key): float(value)
            for key, value in coverage.items()
            if isinstance(value, (int, float))
        }
    )

    counts = results.get("counts", {})

    return {
        "source_type": "llm",
        "experiment": str(display_name),
        "config_name": config_path.stem,
        "config_file": str(config_path),
        "dataset": str(
            results.get("dataset", dataset_name)
        ),
        "model": str(
            results.get("model", model_name)
        ),
        "model_family": model_family(model_name),
        "selected_checkpoint": results.get(
            "selected_checkpoint"
        ),
        "metrics_file": str(metrics_file),
        "per_user_metrics_file": (
            str(per_user_file)
            if per_user_file.is_file()
            else None
        ),
        "evaluated_users": counts.get(
            "evaluated_users"
        ),
        "metrics": combined_metrics,
        "confidence_intervals_95": results.get(
            "confidence_intervals_95",
            {},
        ),
        "diagnostics": results.get(
            "diagnostics",
            {},
        ),
    }


# ---------------------------------------------------------------------------
# Optional baseline loading
# ---------------------------------------------------------------------------

def load_baselines(path: Path) -> list[dict[str, Any]]:
    """
    Load optional RecBole baseline results.

    Expected structure:

    {
      "Fashion": [
        {
          "experiment": "SASRec baseline",
          "model": "SASRec",
          "metrics": {
            "Hit@10": 0.0201,
            "NDCG@10": 0.0123
          }
        }
      ]
    }
    """

    value = read_json(path)

    if not isinstance(value, dict):
        raise ValueError(
            "The baseline file must be a JSON object keyed by dataset."
        )

    baselines: list[dict[str, Any]] = []

    for dataset_name, entries in value.items():
        if isinstance(entries, dict):
            entries = [entries]

        if not isinstance(entries, list):
            raise ValueError(
                f"Baseline entries for {dataset_name} must be a list."
            )

        for position, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Invalid baseline record for {dataset_name}."
                )

            raw_metrics = entry.get("metrics", {})

            if not isinstance(raw_metrics, dict):
                raise ValueError(
                    f"Invalid baseline metrics for {dataset_name}."
                )

            metrics = {
                str(key): float(value)
                for key, value in raw_metrics.items()
                if isinstance(value, (int, float))
            }

            experiment_name = str(
                entry.get(
                    "experiment",
                    entry.get(
                        "model",
                        f"Baseline {position + 1}",
                    ),
                )
            )

            baselines.append(
                {
                    "source_type": "baseline",
                    "experiment": experiment_name,
                    "config_name": None,
                    "config_file": None,
                    "dataset": str(dataset_name),
                    "model": str(
                        entry.get("model", experiment_name)
                    ),
                    "model_family": str(
                        entry.get("model", experiment_name)
                    ),
                    "selected_checkpoint": None,
                    "metrics_file": str(path),
                    "per_user_metrics_file": None,
                    "evaluated_users": entry.get(
                        "evaluated_users"
                    ),
                    "metrics": metrics,
                    "confidence_intervals_95": {},
                    "diagnostics": {},
                }
            )

    return baselines


# ---------------------------------------------------------------------------
# Per-user paired comparison
# ---------------------------------------------------------------------------

def load_per_user_metric(
    path: Path,
    metric_name: str,
) -> dict[int, float]:
    values: dict[int, float] = {}

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

            record_index = int(record["record_index"])
            metrics = record.get("metrics", {})

            if metric_name not in metrics:
                raise ValueError(
                    f"{metric_name} is missing from line "
                    f"{line_number} of {path}"
                )

            if record_index in values:
                raise ValueError(
                    f"Duplicate record_index {record_index} in {path}"
                )

            values[record_index] = float(
                metrics[metric_name]
            )

    return values


def paired_comparison(
    experiment_a: dict[str, Any],
    experiment_b: dict[str, Any],
    primary_metric: str,
) -> dict[str, Any] | None:
    path_a_value = experiment_a.get(
        "per_user_metrics_file"
    )
    path_b_value = experiment_b.get(
        "per_user_metrics_file"
    )

    if not path_a_value or not path_b_value:
        return None

    path_a = Path(path_a_value)
    path_b = Path(path_b_value)

    if not path_a.is_file() or not path_b.is_file():
        return None

    values_a = load_per_user_metric(
        path_a,
        primary_metric,
    )

    values_b = load_per_user_metric(
        path_b,
        primary_metric,
    )

    indices_a = set(values_a.keys())
    indices_b = set(values_b.keys())

    if indices_a != indices_b:
        return {
            "dataset": experiment_a["dataset"],
            "metric": primary_metric,
            "experiment_a": experiment_a["experiment"],
            "experiment_b": experiment_b["experiment"],
            "status": "not_comparable",
            "reason": (
                "The experiments were not evaluated on exactly "
                "the same test records."
            ),
            "users_in_a": len(indices_a),
            "users_in_b": len(indices_b),
            "shared_users": len(
                indices_a & indices_b
            ),
        }

    ordered_indices = sorted(indices_a)

    differences = [
        values_a[index] - values_b[index]
        for index in ordered_indices
    ]

    mean_difference, lower, upper = (
        confidence_interval_95(differences)
    )

    if lower > 0:
        conclusion = (
            f"{experiment_a['experiment']} is higher"
        )
    elif upper < 0:
        conclusion = (
            f"{experiment_b['experiment']} is higher"
        )
    else:
        conclusion = (
            "No clear difference at the 95% confidence level"
        )

    return {
        "dataset": experiment_a["dataset"],
        "metric": primary_metric,
        "experiment_a": experiment_a["experiment"],
        "experiment_b": experiment_b["experiment"],
        "status": "comparable",
        "evaluated_users": len(ordered_indices),
        "mean_a": experiment_a["metrics"].get(
            primary_metric
        ),
        "mean_b": experiment_b["metrics"].get(
            primary_metric
        ),
        "mean_difference_a_minus_b": mean_difference,
        "confidence_interval_95": [
            lower,
            upper,
        ],
        "confidence_interval_excludes_zero": (
            lower > 0 or upper < 0
        ),
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# Ranking and output
# ---------------------------------------------------------------------------

def assign_dataset_ranks(
    experiments: list[dict[str, Any]],
    primary_metric: str,
) -> None:
    datasets = sorted(
        set(
            experiment["dataset"]
            for experiment in experiments
        )
    )

    for dataset_name in datasets:
        dataset_experiments = [
            experiment
            for experiment in experiments
            if experiment["dataset"] == dataset_name
        ]

        ranked = sorted(
            dataset_experiments,
            key=lambda experiment: (
                experiment["metrics"].get(
                    primary_metric,
                    float("-inf"),
                )
            ),
            reverse=True,
        )

        previous_value: float | None = None
        previous_rank = 0

        for position, experiment in enumerate(
            ranked,
            start=1,
        ):
            value = experiment["metrics"].get(
                primary_metric
            )

            if value is None:
                experiment["rank"] = None
                continue

            if previous_value is None or value != previous_value:
                previous_rank = position

            experiment["rank"] = previous_rank
            previous_value = value


def percentage_change(
    new_value: float,
    old_value: float,
) -> float | None:
    if old_value == 0:
        return None

    return (
        (new_value - old_value)
        / abs(old_value)
        * 100.0
    )


def add_baseline_differences(
    experiments: list[dict[str, Any]],
    primary_metric: str,
) -> None:
    datasets = set(
        experiment["dataset"]
        for experiment in experiments
    )

    for dataset_name in datasets:
        dataset_entries = [
            experiment
            for experiment in experiments
            if experiment["dataset"] == dataset_name
        ]

        baselines = [
            experiment
            for experiment in dataset_entries
            if experiment["source_type"] == "baseline"
            and primary_metric in experiment["metrics"]
        ]

        llm_entries = [
            experiment
            for experiment in dataset_entries
            if experiment["source_type"] == "llm"
            and primary_metric in experiment["metrics"]
        ]

        if not baselines:
            continue

        for experiment in llm_entries:
            experiment["baseline_comparisons"] = []

            model_value = experiment["metrics"][
                primary_metric
            ]

            for baseline in baselines:
                baseline_value = baseline["metrics"][
                    primary_metric
                ]

                experiment["baseline_comparisons"].append(
                    {
                        "baseline": baseline["experiment"],
                        "metric": primary_metric,
                        "absolute_difference": (
                            model_value - baseline_value
                        ),
                        "percentage_change": percentage_change(
                            model_value,
                            baseline_value,
                        ),
                    }
                )


def all_metric_names(
    experiments: list[dict[str, Any]],
) -> list[str]:
    metric_names: set[str] = set()

    for experiment in experiments:
        metric_names.update(
            experiment["metrics"].keys()
        )

    metric_order = {
        "Hit": 0,
        "Recall": 1,
        "NDCG": 2,
        "MRR": 3,
        "CatalogueCoverage": 4,
    }

    def sort_key(metric_name: str) -> tuple[int, int, str]:
        if "@" in metric_name:
            prefix, cutoff_text = metric_name.rsplit("@", 1)

            try:
                cutoff = int(cutoff_text)
            except ValueError:
                cutoff = 999999
        else:
            prefix = metric_name
            cutoff = 999999

        return (
            cutoff,
            metric_order.get(prefix, 999),
            metric_name,
        )

    return sorted(metric_names, key=sort_key)


def write_csv_comparison(
    path: Path,
    experiments: list[dict[str, Any]],
    metric_names: list[str],
    primary_metric: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "rank",
        "experiment",
        "source_type",
        "model",
        "evaluated_users",
        "primary_metric",
        "primary_metric_value",
        *metric_names,
        "config_file",
        "metrics_file",
        "selected_checkpoint",
    ]

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        sorted_experiments = sorted(
            experiments,
            key=lambda experiment: (
                experiment["dataset"],
                (
                    experiment["rank"]
                    if experiment["rank"] is not None
                    else 999999
                ),
                experiment["experiment"],
            ),
        )

        for experiment in sorted_experiments:
            row = {
                "dataset": experiment["dataset"],
                "rank": experiment["rank"],
                "experiment": experiment["experiment"],
                "source_type": experiment["source_type"],
                "model": experiment["model"],
                "evaluated_users": experiment[
                    "evaluated_users"
                ],
                "primary_metric": primary_metric,
                "primary_metric_value": (
                    experiment["metrics"].get(
                        primary_metric
                    )
                ),
                "config_file": experiment[
                    "config_file"
                ],
                "metrics_file": experiment[
                    "metrics_file"
                ],
                "selected_checkpoint": experiment[
                    "selected_checkpoint"
                ],
            }

            for metric_name in metric_names:
                row[metric_name] = experiment[
                    "metrics"
                ].get(metric_name)

            writer.writerow(row)

    os.replace(temporary_path, path)


def format_number(value: Any) -> str:
    if value is None:
        return "—"

    if isinstance(value, int):
        return f"{value:,}"

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


def write_markdown_comparison(
    path: Path,
    experiments: list[dict[str, Any]],
    paired_comparisons: list[dict[str, Any]],
    primary_metric: str,
    display_metrics: list[str],
) -> None:
    lines = [
        "# LLM Recommender Experiment Comparison",
        "",
        f"Primary ranking metric: **{primary_metric}**",
        "",
        (
            "Models are ranked separately within each dataset. "
            "Results from different datasets are not directly compared."
        ),
        "",
    ]

    datasets = sorted(
        set(
            experiment["dataset"]
            for experiment in experiments
        )
    )

    for dataset_name in datasets:
        dataset_experiments = sorted(
            [
                experiment
                for experiment in experiments
                if experiment["dataset"] == dataset_name
            ],
            key=lambda experiment: (
                experiment["rank"]
                if experiment["rank"] is not None
                else 999999
            ),
        )

        available_display_metrics = [
            metric
            for metric in display_metrics
            if any(
                metric in experiment["metrics"]
                for experiment in dataset_experiments
            )
        ]

        lines.extend(
            [
                f"## {dataset_name}",
                "",
            ]
        )

        headers = [
            "Rank",
            "Experiment",
            "Type",
            "Users",
            *available_display_metrics,
        ]

        lines.append(
            "| " + " | ".join(headers) + " |"
        )

        lines.append(
            "| "
            + " | ".join(
                ["---"] * len(headers)
            )
            + " |"
        )

        for experiment in dataset_experiments:
            row = [
                format_number(experiment["rank"]),
                experiment["experiment"],
                experiment["source_type"],
                format_number(
                    experiment["evaluated_users"]
                ),
            ]

            row.extend(
                format_number(
                    experiment["metrics"].get(metric)
                )
                for metric in available_display_metrics
            )

            lines.append(
                "| " + " | ".join(row) + " |"
            )

        lines.append("")

        best = next(
            (
                experiment
                for experiment in dataset_experiments
                if experiment["rank"] == 1
                and primary_metric
                in experiment["metrics"]
            ),
            None,
        )

        if best is not None:
            lines.append(
                f"Best result by {primary_metric}: "
                f"**{best['experiment']}** "
                f"({best['metrics'][primary_metric]:.6f})."
            )
            lines.append("")

        dataset_pairs = [
            comparison
            for comparison in paired_comparisons
            if comparison["dataset"] == dataset_name
            and comparison["status"] == "comparable"
        ]

        if dataset_pairs:
            lines.extend(
                [
                    "### Paired comparisons",
                    "",
                    (
                        "The difference is calculated as Experiment A "
                        "minus Experiment B on the same test users."
                    ),
                    "",
                    (
                        "| Experiment A | Experiment B | Metric | "
                        "Difference | 95% CI | Conclusion |"
                    ),
                    "| --- | --- | --- | ---: | --- | --- |",
                ]
            )

            for comparison in dataset_pairs:
                lower, upper = comparison[
                    "confidence_interval_95"
                ]

                lines.append(
                    "| "
                    f"{comparison['experiment_a']} | "
                    f"{comparison['experiment_b']} | "
                    f"{comparison['metric']} | "
                    f"{comparison['mean_difference_a_minus_b']:.6f} | "
                    f"[{lower:.6f}, {upper:.6f}] | "
                    f"{comparison['conclusion']} |"
                )

            lines.append("")

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))
        file.write("\n")

    os.replace(temporary_path, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Step 07 results across multiple QLoRA "
            "recommender experiments."
        )
    )

    parser.add_argument(
        "configs",
        type=Path,
        nargs="+",
        help="Two or more LLM experiment YAML files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/llm_comparison"),
        help=(
            "Directory in which comparison files will be saved."
        ),
    )

    parser.add_argument(
        "--primary-metric",
        type=str,
        default="NDCG@10",
        help="Metric used to rank models within each dataset.",
    )

    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing SASRec, BERT4Rec or "
            "other baseline results."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if len(args.configs) < 2:
        raise ValueError(
            "Provide at least two experiment configurations."
        )

    print("=" * 70)
    print("COMPARING LLM RECOMMENDER EXPERIMENTS")
    print("=" * 70)
    print(f"Primary metric: {args.primary_metric}")
    print()

    experiments: list[dict[str, Any]] = []

    for config_path in args.configs:
        print(f"Loading: {config_path}")

        experiment = load_experiment(config_path)
        experiments.append(experiment)

        print(
            f"  Dataset:    {experiment['dataset']}\n"
            f"  Experiment: {experiment['experiment']}\n"
            f"  Users:      {experiment['evaluated_users']}"
        )

    if args.baseline_file is not None:
        baseline_path = args.baseline_file.resolve()

        if not baseline_path.is_file():
            raise FileNotFoundError(
                f"Baseline file not found: {baseline_path}"
            )

        baselines = load_baselines(baseline_path)
        experiments.extend(baselines)

        print(
            f"Loaded {len(baselines)} baseline result(s)."
        )

    missing_primary_metric = [
        (
            experiment["dataset"],
            experiment["experiment"],
        )
        for experiment in experiments
        if args.primary_metric
        not in experiment["metrics"]
    ]

    if missing_primary_metric:
        formatted = "\n".join(
            f"  - {dataset}: {experiment}"
            for dataset, experiment
            in missing_primary_metric
        )

        raise ValueError(
            f"The following experiments do not contain "
            f"{args.primary_metric}:\n{formatted}"
        )

    assign_dataset_ranks(
        experiments=experiments,
        primary_metric=args.primary_metric,
    )

    add_baseline_differences(
        experiments=experiments,
        primary_metric=args.primary_metric,
    )

    paired_comparisons: list[dict[str, Any]] = []

    llm_datasets = sorted(
        set(
            experiment["dataset"]
            for experiment in experiments
            if experiment["source_type"] == "llm"
        )
    )

    print()
    print("Calculating paired comparisons...")

    for dataset_name in llm_datasets:
        dataset_llm_experiments = [
            experiment
            for experiment in experiments
            if experiment["dataset"] == dataset_name
            and experiment["source_type"] == "llm"
        ]

        for experiment_a, experiment_b in itertools.combinations(
            dataset_llm_experiments,
            2,
        ):
            comparison = paired_comparison(
                experiment_a=experiment_a,
                experiment_b=experiment_b,
                primary_metric=args.primary_metric,
            )

            if comparison is not None:
                paired_comparisons.append(comparison)

    metric_names = all_metric_names(experiments)

    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    json_output = (
        output_directory
        / "comparison_results.json"
    )

    csv_output = (
        output_directory
        / "comparison_results.csv"
    )

    markdown_output = (
        output_directory
        / "comparison_table.md"
    )

    comparison_result = {
        "primary_metric": args.primary_metric,
        "experiment_count": len(experiments),
        "datasets": sorted(
            set(
                experiment["dataset"]
                for experiment in experiments
            )
        ),
        "metric_names": metric_names,
        "experiments": sorted(
            experiments,
            key=lambda experiment: (
                experiment["dataset"],
                (
                    experiment["rank"]
                    if experiment["rank"] is not None
                    else 999999
                ),
            ),
        ),
        "paired_comparisons": paired_comparisons,
    }

    atomic_write_json(
        json_output,
        comparison_result,
    )

    write_csv_comparison(
        path=csv_output,
        experiments=experiments,
        metric_names=metric_names,
        primary_metric=args.primary_metric,
    )

    display_metrics = [
        "Hit@10",
        "Recall@10",
        "NDCG@10",
        "MRR@10",
        "CatalogueCoverage@10",
    ]

    write_markdown_comparison(
        path=markdown_output,
        experiments=experiments,
        paired_comparisons=paired_comparisons,
        primary_metric=args.primary_metric,
        display_metrics=display_metrics,
    )

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    datasets = sorted(
        set(
            experiment["dataset"]
            for experiment in experiments
        )
    )

    for dataset_name in datasets:
        ranked = sorted(
            [
                experiment
                for experiment in experiments
                if experiment["dataset"] == dataset_name
            ],
            key=lambda experiment: (
                experiment["rank"]
                if experiment["rank"] is not None
                else 999999
            ),
        )

        print()
        print(dataset_name)

        for experiment in ranked:
            value = experiment["metrics"][
                args.primary_metric
            ]

            print(
                f"  {experiment['rank']}. "
                f"{experiment['experiment']}: "
                f"{args.primary_metric}={value:.6f}"
            )

    print()
    print(f"JSON results:     {json_output}")
    print(f"CSV results:      {csv_output}")
    print(f"Markdown table:   {markdown_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()