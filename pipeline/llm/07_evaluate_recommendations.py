"""
Step 07: Evaluate final LLM test recommendations.

The script:

1. Loads the test split created in Step 01.
2. Loads the recommendations generated in Step 06.
3. Matches each prediction to the authoritative test record.
4. Validates catalogue membership, duplicate recommendations, histories,
   targets, and record completeness.
5. Calculates:
      - Hit@K
      - Recall@K
      - NDCG@K
      - MRR@K
      - Catalogue Coverage@K
6. Saves aggregate and per-user evaluation results.

For leave-one-out next-item evaluation, each user has exactly one relevant
test item. Therefore Hit@K and Recall@K are numerically equal, but both are
reported for compatibility with recommender-system literature.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import yaml
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Configuration and path helpers
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

    return "dataset"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


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


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

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

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} of {path} is not a JSON object."
                )

            records.append(record)

    return records


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")

    os.replace(temporary_path, path)


# ---------------------------------------------------------------------------
# Dataset schema helpers
# ---------------------------------------------------------------------------

def first_present(
    record: dict[str, Any],
    keys: Iterable[str],
    required: bool = True,
) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]

    if required:
        raise KeyError(
            f"None of the required fields were found: {list(keys)}. "
            f"Available fields: {sorted(record.keys())}"
        )

    return None


def normalise_item_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, tuple):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return []

        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)

                if isinstance(decoded, list):
                    return normalise_item_list(decoded)

            except json.JSONDecodeError:
                pass

        if "," in stripped:
            return [
                item.strip()
                for item in stripped.split(",")
                if item.strip()
            ]

        return stripped.split()

    raise TypeError(
        f"Expected an item list or string, received "
        f"{type(value).__name__}."
    )


def test_record_values(
    record: dict[str, Any],
) -> tuple[str, list[str], str]:
    user_id = str(
        first_present(
            record,
            ["user_id", "user", "uid"],
        )
    ).strip()

    history = normalise_item_list(
        first_present(
            record,
            [
                "history",
                "history_item_ids",
                "item_history",
                "input_items",
                "sequence",
            ],
        )
    )

    target = str(
        first_present(
            record,
            [
                "target",
                "target_item_id",
                "next_item_id",
                "label",
                "test_item",
            ],
        )
    ).strip()

    if not user_id:
        raise ValueError("A test record has an empty user ID.")

    if not history:
        raise ValueError(
            f"User {user_id} has an empty test history."
        )

    if not target:
        raise ValueError(
            f"User {user_id} has an empty test target."
        )

    return user_id, history, target


# ---------------------------------------------------------------------------
# Catalogue loading
# ---------------------------------------------------------------------------

def extract_catalogue_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        result: list[str] = []

        for entry in value:
            if isinstance(entry, str):
                result.append(entry)

            elif isinstance(entry, dict):
                item_id = first_present(
                    entry,
                    ["item_id", "asin", "id"],
                    required=False,
                )

                if item_id is not None:
                    result.append(str(item_id))

        return result

    if isinstance(value, dict):
        for key in [
            "item_ids",
            "catalogue",
            "catalog",
            "items",
            "records",
        ]:
            if key in value:
                result = extract_catalogue_ids(value[key])

                if result:
                    return result

        if value and all(
            isinstance(record, dict)
            for record in value.values()
        ):
            return [
                str(item_id)
                for item_id in value.keys()
            ]

    raise ValueError(
        "Could not identify item IDs in the catalogue file."
    )


def load_catalogue(path: Path) -> list[str]:
    catalogue = extract_catalogue_ids(read_json(path))

    unique_catalogue = list(
        dict.fromkeys(
            item_id.strip()
            for item_id in catalogue
            if item_id.strip()
        )
    )

    if not unique_catalogue:
        raise ValueError(
            f"The item catalogue is empty: {path}"
        )

    return unique_catalogue


# ---------------------------------------------------------------------------
# Metric accumulation
# ---------------------------------------------------------------------------

class MetricAccumulator:
    """
    Incrementally tracks a mean, sample variance and 95% confidence interval.
    """

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_squared = 0.0

    def add(self, value: float) -> None:
        numeric_value = float(value)

        self.count += 1
        self.total += numeric_value
        self.total_squared += numeric_value ** 2

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0

        return self.total / self.count

    @property
    def sample_variance(self) -> float:
        if self.count <= 1:
            return 0.0

        numerator = (
            self.total_squared
            - (self.total ** 2 / self.count)
        )

        return max(
            0.0,
            numerator / (self.count - 1),
        )

    @property
    def standard_error(self) -> float:
        if self.count == 0:
            return 0.0

        return math.sqrt(
            self.sample_variance / self.count
        )

    def confidence_interval_95(self) -> list[float]:
        margin = 1.96 * self.standard_error

        lower = max(0.0, self.mean - margin)
        upper = min(1.0, self.mean + margin)

        return [lower, upper]


def calculate_rank(
    target: str,
    recommendations: list[str],
) -> int | None:
    """
    Return the one-based target rank, or None if the target was not ranked.
    """

    try:
        return recommendations.index(target) + 1
    except ValueError:
        return None


def user_metrics_at_k(
    rank: int | None,
    k: int,
) -> dict[str, float]:
    if rank is None or rank > k:
        return {
            "hit": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
            "mrr": 0.0,
        }

    return {
        "hit": 1.0,
        "recall": 1.0,
        "ndcg": 1.0 / math.log2(rank + 1),
        "mrr": 1.0 / rank,
    }


# ---------------------------------------------------------------------------
# Prediction loading and validation
# ---------------------------------------------------------------------------

def load_predictions(
    path: Path,
) -> dict[int, dict[str, Any]]:
    predictions: dict[int, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}. "
                    f"Step 06 may have been interrupted while writing."
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Prediction line {line_number} is not a JSON object."
                )

            if "record_index" not in record:
                raise ValueError(
                    f"Prediction line {line_number} does not contain "
                    f"'record_index'."
                )

            record_index = int(record["record_index"])

            if record_index in predictions:
                raise ValueError(
                    f"Duplicate prediction for record_index "
                    f"{record_index}."
                )

            predictions[record_index] = record

    return predictions


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate QLoRA next-item recommendations on the final "
            "test split."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment YAML configuration.",
    )

    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Allow evaluation when Step 06 contains only a subset of "
            "test users. Use this only for debugging."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config_path = args.config.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    config = load_yaml(config_path)
    repository_root = find_repository_root(config_path)

    dataset_name = dataset_name_from_config(config)

    model_name = str(
        nested_value(
            config,
            "model.name_or_path",
            default="unknown",
        )
    )

    output_directory_value = nested_value(
        config,
        "training.output_dir",
    )

    if not output_directory_value:
        raise ValueError(
            "The YAML file must contain training.output_dir."
        )

    output_directory = resolve_path(
        output_directory_value,
        repository_root,
    )

    test_file_value = nested_value(
        config,
        "data.test_split_file",
    )

    catalogue_file_value = nested_value(
        config,
        "data.catalogue_file",
    )

    if not test_file_value:
        raise ValueError(
            "The YAML file must contain data.test_split_file."
        )

    if not catalogue_file_value:
        raise ValueError(
            "The YAML file must contain data.catalogue_file."
        )

    test_file = resolve_path(
        test_file_value,
        repository_root,
    )

    catalogue_file = resolve_path(
        catalogue_file_value,
        repository_root,
    )

    predictions_file_value = nested_value(
        config,
        "generation.output_file",
        default=(
            output_directory
            / "test_predictions"
            / "test_predictions.jsonl"
        ),
    )

    predictions_file = resolve_path(
        predictions_file_value,
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

    metrics_file = resolve_path(
        metrics_file_value,
        repository_root,
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

    per_user_file = resolve_path(
        per_user_file_value,
        repository_root,
    )

    generation_summary_file = (
        predictions_file.parent
        / "generation_summary.json"
    )

    required_files = [
        test_file,
        catalogue_file,
        predictions_file,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            f"Required Step 07 inputs are missing:\n{formatted}"
        )

    raw_cutoffs = nested_value(
        config,
        "evaluation.cutoffs",
        default=[1, 5, 10],
    )

    if not isinstance(raw_cutoffs, list):
        raise ValueError(
            "evaluation.cutoffs must be a YAML list."
        )

    cutoffs = sorted(
        set(int(k) for k in raw_cutoffs)
    )

    if not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError(
            "All evaluation cutoffs must be positive."
        )

    exclude_history_items = bool(
        nested_value(
            config,
            "generation.exclude_history_items",
            default=True,
        )
    )

    print("=" * 70)
    print("EVALUATING TEST RECOMMENDATIONS")
    print("=" * 70)
    print(f"Dataset:          {dataset_name}")
    print(f"Model:            {model_name}")
    print(f"Test split:       {test_file}")
    print(f"Predictions:      {predictions_file}")
    print(f"Catalogue:        {catalogue_file}")
    print(f"Cutoffs:          {cutoffs}")
    print(f"Metrics output:   {metrics_file}")
    print(f"Per-user output:  {per_user_file}")
    print()

    print("Loading authoritative test split...")
    test_records = read_jsonl(test_file)
    print(f"Test users: {len(test_records):,}")

    print("Loading predictions...")
    predictions = load_predictions(predictions_file)
    print(f"Prediction records: {len(predictions):,}")

    print("Loading catalogue...")
    catalogue = load_catalogue(catalogue_file)
    catalogue_set = set(catalogue)
    print(f"Catalogue items: {len(catalogue):,}")

    expected_indices = set(range(len(test_records)))
    prediction_indices = set(predictions.keys())

    missing_indices = sorted(
        expected_indices - prediction_indices
    )

    unexpected_indices = sorted(
        prediction_indices - expected_indices
    )

    if unexpected_indices:
        raise ValueError(
            f"Predictions contain {len(unexpected_indices):,} "
            f"unexpected record indices. First indices: "
            f"{unexpected_indices[:10]}"
        )

    if missing_indices and not args.allow_incomplete:
        raise ValueError(
            f"Step 06 predictions are incomplete. "
            f"{len(missing_indices):,} of "
            f"{len(test_records):,} test users are missing.\n"
            f"First missing indices: {missing_indices[:10]}\n"
            f"Resume Step 06, or use --allow-incomplete only for a "
            f"debugging run."
        )

    evaluation_indices = sorted(
        expected_indices & prediction_indices
    )

    if not evaluation_indices:
        raise ValueError(
            "There are no test predictions to evaluate."
        )

    if missing_indices:
        print(
            f"Warning: incomplete debug evaluation. "
            f"{len(missing_indices):,} test users are missing."
        )

    accumulators: dict[str, MetricAccumulator] = {}

    for k in cutoffs:
        for metric_name in [
            "Hit",
            "Recall",
            "NDCG",
            "MRR",
        ]:
            accumulators[
                f"{metric_name}@{k}"
            ] = MetricAccumulator()

    coverage_items: dict[int, set[str]] = {
        k: set()
        for k in cutoffs
    }

    recommendation_count = MetricAccumulator()

    target_found_count = 0
    duplicate_row_count = 0
    invalid_item_count = 0
    history_overlap_count = 0
    target_in_history_count = 0
    short_row_count = 0

    maximum_cutoff = max(cutoffs)

    per_user_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_per_user_file = per_user_file.with_suffix(
        per_user_file.suffix + ".tmp"
    )

    with temporary_per_user_file.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for record_index in tqdm(
            evaluation_indices,
            desc="Evaluating users",
            unit="user",
        ):
            test_record = test_records[record_index]
            prediction_record = predictions[record_index]

            user_id, history, target = test_record_values(
                test_record
            )

            predicted_user_id = str(
                prediction_record.get("user_id", "")
            ).strip()

            if predicted_user_id != user_id:
                raise ValueError(
                    f"User mismatch at record_index {record_index}: "
                    f"test user={user_id!r}, "
                    f"prediction user={predicted_user_id!r}."
                )

            prediction_target = str(
                prediction_record.get("target", "")
            ).strip()

            if (
                prediction_target
                and prediction_target != target
            ):
                raise ValueError(
                    f"Target mismatch at record_index {record_index}: "
                    f"test target={target!r}, "
                    f"prediction target={prediction_target!r}."
                )

            prediction_history_value = (
                prediction_record.get("history")
            )

            if prediction_history_value is not None:
                prediction_history = normalise_item_list(
                    prediction_history_value
                )

                if prediction_history != history:
                    raise ValueError(
                        f"History mismatch at record_index "
                        f"{record_index}."
                    )

            recommendations = normalise_item_list(
                prediction_record.get(
                    "recommendations",
                    [],
                )
            )

            if len(recommendations) < maximum_cutoff:
                short_row_count += 1

            recommendation_count.add(
                len(recommendations)
            )

            if len(recommendations) != len(
                set(recommendations)
            ):
                duplicate_row_count += 1

                duplicate_items = [
                    item_id
                    for position, item_id
                    in enumerate(recommendations)
                    if item_id
                    in recommendations[:position]
                ]

                raise ValueError(
                    f"Duplicate recommendations for user "
                    f"{user_id}: {duplicate_items[:10]}"
                )

            invalid_items = [
                item_id
                for item_id in recommendations
                if item_id not in catalogue_set
            ]

            if invalid_items:
                invalid_item_count += len(invalid_items)

                raise ValueError(
                    f"User {user_id} has recommendations outside "
                    f"the catalogue: {invalid_items[:10]}"
                )

            history_set = set(history)

            history_overlaps = [
                item_id
                for item_id in recommendations
                if item_id in history_set
            ]

            if history_overlaps:
                history_overlap_count += len(
                    history_overlaps
                )

                if exclude_history_items:
                    raise ValueError(
                        f"User {user_id} received items already in "
                        f"their history: {history_overlaps[:10]}"
                    )

            if target in history_set:
                # RecBole masks previously interacted items during full-sort
                # evaluation, so a repeated target is retained as a guaranteed miss.
                target_in_history_count += 1

            rank = calculate_rank(
                target=target,
                recommendations=recommendations,
            )

            if rank is not None:
                target_found_count += 1

            per_user_metric_values: dict[str, float] = {}

            for k in cutoffs:
                user_values = user_metrics_at_k(
                    rank=rank,
                    k=k,
                )

                for metric_name, value in (
                    ("Hit", user_values["hit"]),
                    ("Recall", user_values["recall"]),
                    ("NDCG", user_values["ndcg"]),
                    ("MRR", user_values["mrr"]),
                ):
                    metric_key = f"{metric_name}@{k}"

                    accumulators[metric_key].add(value)
                    per_user_metric_values[
                        metric_key
                    ] = value

                coverage_items[k].update(
                    recommendations[:k]
                )

            per_user_result = {
                "record_index": record_index,
                "user_id": user_id,
                "target": target,
                "target_rank": rank,
                "recommendation_count": len(
                    recommendations
                ),
                "recommendations": recommendations,
                "metrics": per_user_metric_values,
            }

            output_file.write(
                json.dumps(
                    per_user_result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    os.replace(
        temporary_per_user_file,
        per_user_file,
    )

    metrics = {
        metric_name: accumulator.mean
        for metric_name, accumulator
        in accumulators.items()
    }

    confidence_intervals = {
        metric_name: (
            accumulator.confidence_interval_95()
        )
        for metric_name, accumulator
        in accumulators.items()
    }

    coverage_metrics = {
        f"CatalogueCoverage@{k}": (
            len(coverage_items[k])
            / len(catalogue)
        )
        for k in cutoffs
    }

    unique_recommendation_counts = {
        f"UniqueRecommendedItems@{k}": len(
            coverage_items[k]
        )
        for k in cutoffs
    }

    selected_checkpoint = None

    if generation_summary_file.is_file():
        generation_summary = read_json(
            generation_summary_file
        )

        if isinstance(generation_summary, dict):
            selected_checkpoint = (
                generation_summary.get(
                    "selected_checkpoint"
                )
            )

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "selected_checkpoint": selected_checkpoint,
        "evaluation_protocol": {
            "name": "leave-one-out next-item prediction",
            "test_history": "items 1 through n-1",
            "test_target": "item n",
            "candidate_strategy": (
                "full-catalogue constrained generation"
            ),
            "history_items_excluded": (
                exclude_history_items
            ),
            "relevant_items_per_user": 1,
            "cutoffs": cutoffs,
        },
        "input_files": {
            "config": str(config_path),
            "test_split": str(test_file),
            "predictions": str(predictions_file),
            "catalogue": str(catalogue_file),
        },
        "counts": {
            "test_records": len(test_records),
            "prediction_records": len(predictions),
            "evaluated_users": len(
                evaluation_indices
            ),
            "missing_prediction_records": len(
                missing_indices
            ),
            "catalogue_items": len(catalogue),
            "targets_found_in_full_generated_list": (
                target_found_count
            ),
        },
        "metrics": metrics,
        "confidence_intervals_95": (
            confidence_intervals
        ),
        "coverage": coverage_metrics,
        "unique_recommendation_counts": (
            unique_recommendation_counts
        ),
        "diagnostics": {
            "average_recommendation_count": (
                recommendation_count.mean
            ),
            "rows_shorter_than_maximum_cutoff": (
                short_row_count
            ),
            "duplicate_recommendation_rows": (
                duplicate_row_count
            ),
            "invalid_recommendation_items": (
                invalid_item_count
            ),
            "history_overlap_items": (
                history_overlap_count
            ),
            "targets_already_in_history": (
                target_in_history_count
            ),
            "incomplete_debug_evaluation": bool(
                missing_indices
            ),
        },
    }

    atomic_write_json(
        metrics_file,
        result,
    )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"Evaluated users: {len(evaluation_indices):,}")
    print()

    for k in cutoffs:
        print(f"Metrics at K={k}")
        print(f"  Hit@{k}:    {metrics[f'Hit@{k}']:.6f}")
        print(
            f"  Recall@{k}: "
            f"{metrics[f'Recall@{k}']:.6f}"
        )
        print(
            f"  NDCG@{k}:   "
            f"{metrics[f'NDCG@{k}']:.6f}"
        )
        print(f"  MRR@{k}:    {metrics[f'MRR@{k}']:.6f}")
        print(
            f"  Coverage@{k}: "
            f"{coverage_metrics[f'CatalogueCoverage@{k}']:.6f}"
        )
        print()

    print(
        f"Average recommendation count: "
        f"{recommendation_count.mean:.2f}"
    )
    print(
        f"Rows with fewer than {maximum_cutoff} recommendations: "
        f"{short_row_count:,}"
    )
    print()
    print(f"Aggregate metrics: {metrics_file}")
    print(f"Per-user metrics:  {per_user_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()