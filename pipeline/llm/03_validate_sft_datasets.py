"""
Validate the temporal splits and QLoRA SFT datasets.

This script verifies that:

1. Every user appears in train, validation, and test.
2. Validation history equals the user's training sequence.
3. Test history equals the training sequence plus the real validation item.
4. Validation and test targets occur at the correct chronological positions.
5. SFT training examples match every expected prefix-to-next-item pair.
6. Validation SFT examples contain the correct held-out target.
7. Targets are not accidentally inserted into prompts.
8. Every referenced item belongs to the dataset catalogue.
9. Product metadata is available, except explicitly permitted missing titles.
"""

from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

from common import (
    build_recommendation_messages,
    clean_text,
    iter_jsonl,
    load_metadata_lookup,
    read_json,
    require_fields,
    resolve_project_path,
    write_json,
)


MISSING = object()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate temporal splits and QLoRA SFT datasets."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, for example Fashion or Video Games.",
    )

    parser.add_argument(
        "--split-dir",
        required=True,
        help="Directory containing the Step 01 outputs.",
    )

    parser.add_argument(
        "--sft-dir",
        required=True,
        help="Directory containing the Step 02 outputs.",
    )

    parser.add_argument(
        "--metadata-file",
        required=True,
        help="Metadata lookup used when building the SFT datasets.",
    )

    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="Permit items whose metadata contains no usable title.",
    )

    parser.add_argument(
        "--max-errors",
        type=int,
        default=100,
        help="Maximum number of detailed errors stored in the report.",
    )

    return parser.parse_args()


class ValidationResults:
    """Collect validation checks, errors, and warnings."""

    def __init__(self, max_errors: int = 100) -> None:
        self.max_errors = max_errors
        self.checks_performed = 0
        self.total_errors = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check(
        self,
        condition: bool,
        message: str,
    ) -> bool:
        self.checks_performed += 1

        if condition:
            return True

        self.total_errors += 1

        if len(self.errors) < self.max_errors:
            self.errors.append(message)

        return False

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def passed(self) -> bool:
        return self.total_errors == 0


def has_usable_title(metadata: dict[str, Any] | None) -> bool:
    if not metadata:
        return False

    for field in ("title", "name", "product_title"):
        if clean_text(metadata.get(field)):
            return True

    return False


def timestamps_are_ordered(timestamps: list[Any]) -> bool:
    try:
        numeric = [float(timestamp) for timestamp in timestamps]
    except (TypeError, ValueError):
        return False

    return all(
        numeric[index] <= numeric[index + 1]
        for index in range(len(numeric) - 1)
    )


def load_catalogue(
    catalogue_path: Path,
    results: ValidationResults,
) -> set[str]:
    catalogue = read_json(catalogue_path)

    results.check(
        isinstance(catalogue, dict),
        "Item catalogue is not a JSON object.",
    )

    if not isinstance(catalogue, dict):
        return set()

    items = catalogue.get("items")

    results.check(
        isinstance(items, list),
        "Item catalogue does not contain an 'items' list.",
    )

    if not isinstance(items, list):
        return set()

    item_ids: set[str] = set()

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            results.check(
                False,
                f"Catalogue entry {index} is not a JSON object.",
            )
            continue

        item_id = item.get("item_id")

        if not results.check(
            item_id is not None and str(item_id).strip() != "",
            f"Catalogue entry {index} has no item ID.",
        ):
            continue

        item_id = str(item_id)

        results.check(
            item_id not in item_ids,
            f"Duplicate catalogue item: {item_id}",
        )

        item_ids.add(item_id)

    declared_count = catalogue.get("num_items")

    if declared_count is not None:
        results.check(
            declared_count == len(item_ids),
            "Catalogue num_items does not match the actual item count.",
        )

    return item_ids


def load_unique_user_records(
    path: Path,
    split_name: str,
    results: ValidationResults,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    for record_number, record in enumerate(
        iter_jsonl(path),
        start=1,
    ):
        if "user_id" not in record:
            results.check(
                False,
                f"{split_name} record {record_number} has no user_id.",
            )
            continue

        user_id = str(record["user_id"])

        results.check(
            user_id not in records,
            f"Duplicate user {user_id} in {split_name}.",
        )

        records[user_id] = record

    return records


def validate_temporal_splits(
    train_records: dict[str, dict[str, Any]],
    validation_records: dict[str, dict[str, Any]],
    test_records: dict[str, dict[str, Any]],
    catalogue_item_ids: set[str],
    results: ValidationResults,
) -> dict[str, int]:
    train_users = set(train_records)
    validation_users = set(validation_records)
    test_users = set(test_records)

    results.check(
        train_users == validation_users,
        "Training and validation user sets do not match.",
    )

    results.check(
        train_users == test_users,
        "Training and test user sets do not match.",
    )

    total_train_interactions = 0
    expected_training_examples = 0

    for user_id in sorted(train_users):
        train_record = train_records[user_id]
        validation_record = validation_records.get(user_id)
        test_record = test_records.get(user_id)

        if validation_record is None or test_record is None:
            continue

        try:
            require_fields(
                train_record,
                ["item_ids", "timestamps"],
                context=f"train user {user_id}",
            )
            require_fields(
                validation_record,
                [
                    "history_item_ids",
                    "history_timestamps",
                    "target_item_id",
                    "target_timestamp",
                ],
                context=f"validation user {user_id}",
            )
            require_fields(
                test_record,
                [
                    "history_item_ids",
                    "history_timestamps",
                    "target_item_id",
                    "target_timestamp",
                ],
                context=f"test user {user_id}",
            )
        except ValueError as error:
            results.check(False, str(error))
            continue

        train_items = [
            str(item_id) for item_id in train_record["item_ids"]
        ]
        train_timestamps = train_record["timestamps"]

        validation_history = [
            str(item_id)
            for item_id in validation_record["history_item_ids"]
        ]
        validation_timestamps = validation_record["history_timestamps"]
        validation_target = str(
            validation_record["target_item_id"]
        )

        test_history = [
            str(item_id)
            for item_id in test_record["history_item_ids"]
        ]
        test_timestamps = test_record["history_timestamps"]
        test_target = str(test_record["target_item_id"])

        results.check(
            len(train_items) == len(train_timestamps),
            f"Train item/timestamp count mismatch for {user_id}.",
        )

        results.check(
            len(validation_history) == len(validation_timestamps),
            f"Validation item/timestamp count mismatch for {user_id}.",
        )

        results.check(
            len(test_history) == len(test_timestamps),
            f"Test item/timestamp count mismatch for {user_id}.",
        )

        results.check(
            len(train_items) >= 2,
            f"User {user_id} has fewer than two training items.",
        )

        results.check(
            timestamps_are_ordered(train_timestamps),
            f"Training timestamps are not ordered for {user_id}.",
        )

        results.check(
            timestamps_are_ordered(validation_timestamps),
            f"Validation history timestamps are not ordered for {user_id}.",
        )

        results.check(
            timestamps_are_ordered(test_timestamps),
            f"Test history timestamps are not ordered for {user_id}.",
        )

        results.check(
            validation_history == train_items,
            f"Validation history does not equal the training sequence "
            f"for {user_id}.",
        )

        results.check(
            test_history == train_items + [validation_target],
            f"Test history does not equal training history plus the real "
            f"validation target for {user_id}.",
        )

        if train_timestamps:
            results.check(
                float(validation_record["target_timestamp"])
                >= float(train_timestamps[-1]),
                f"Validation target occurs before training for {user_id}.",
            )

        results.check(
            float(test_record["target_timestamp"])
            >= float(validation_record["target_timestamp"]),
            f"Test target occurs before validation for {user_id}.",
        )

        referenced_items = (
            train_items
            + [validation_target]
            + [test_target]
        )

        for item_id in referenced_items:
            results.check(
                item_id in catalogue_item_ids,
                f"Unknown catalogue item {item_id} for user {user_id}.",
            )

        total_train_interactions += len(train_items)
        expected_training_examples += len(train_items) - 1

    return {
        "users": len(train_users),
        "train_interactions": total_train_interactions,
        "expected_training_examples": expected_training_examples,
        "validation_examples": len(validation_users),
        "test_examples": len(test_users),
    }


def expected_training_examples(
    train_records: dict[str, dict[str, Any]],
    metadata_lookup: dict[str, dict[str, Any]],
    max_history_length: int,
) -> Iterator[dict[str, Any]]:
    for user_id in sorted(train_records):
        record = train_records[user_id]

        item_ids = [
            str(item_id) for item_id in record["item_ids"]
        ]
        timestamps = record["timestamps"]

        for target_position in range(1, len(item_ids)):
            complete_history = item_ids[:target_position]
            complete_timestamps = timestamps[:target_position]

            model_history = complete_history[-max_history_length:]
            model_timestamps = (
                complete_timestamps[-max_history_length:]
            )

            target_item_id = item_ids[target_position]

            messages = build_recommendation_messages(
                history_item_ids=model_history,
                metadata_lookup=metadata_lookup,
                max_history_length=max_history_length,
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": target_item_id,
                }
            )

            yield {
                "example_id": f"{user_id}:train:{target_position}",
                "user_id": user_id,
                "split": "train",
                "history_item_ids": model_history,
                "history_timestamps": model_timestamps,
                "target_item_id": target_item_id,
                "target_timestamp": timestamps[target_position],
                "messages": messages,
            }


def expected_validation_examples(
    validation_records: dict[str, dict[str, Any]],
    metadata_lookup: dict[str, dict[str, Any]],
    max_history_length: int,
) -> Iterator[dict[str, Any]]:
    for user_id in sorted(validation_records):
        record = validation_records[user_id]

        complete_history = [
            str(item_id)
            for item_id in record["history_item_ids"]
        ]
        complete_timestamps = record["history_timestamps"]

        model_history = complete_history[-max_history_length:]
        model_timestamps = (
            complete_timestamps[-max_history_length:]
        )
        target_item_id = str(record["target_item_id"])

        messages = build_recommendation_messages(
            history_item_ids=model_history,
            metadata_lookup=metadata_lookup,
            max_history_length=max_history_length,
        )

        messages.append(
            {
                "role": "assistant",
                "content": target_item_id,
            }
        )

        yield {
            "example_id": f"{user_id}:validation",
            "user_id": user_id,
            "split": "validation",
            "history_item_ids": model_history,
            "history_timestamps": model_timestamps,
            "target_item_id": target_item_id,
            "target_timestamp": record["target_timestamp"],
            "messages": messages,
        }


def compare_sft_records(
    expected_path_name: str,
    expected_records: Iterator[dict[str, Any]],
    actual_path: Path,
    results: ValidationResults,
) -> int:
    fields_to_compare = [
        "example_id",
        "user_id",
        "split",
        "history_item_ids",
        "history_timestamps",
        "target_item_id",
        "target_timestamp",
        "messages",
    ]

    actual_records = iter_jsonl(actual_path)
    record_count = 0

    for position, pair in enumerate(
        zip_longest(
            expected_records,
            actual_records,
            fillvalue=MISSING,
        ),
        start=1,
    ):
        expected, actual = pair

        if expected is MISSING:
            results.check(
                False,
                f"{expected_path_name} contains an unexpected extra "
                f"record at position {position}.",
            )
            continue

        if actual is MISSING:
            results.check(
                False,
                f"{expected_path_name} is missing the expected record "
                f"at position {position}.",
            )
            continue

        record_count += 1

        for field in fields_to_compare:
            results.check(
                actual.get(field) == expected.get(field),
                f"{expected_path_name} record {position} has an "
                f"incorrect '{field}' value.",
            )

        messages = actual.get("messages")

        results.check(
            isinstance(messages, list) and len(messages) == 3,
            f"{expected_path_name} record {position} must contain "
            "system, user, and assistant messages.",
        )

        if isinstance(messages, list) and len(messages) == 3:
            roles = [
                message.get("role")
                for message in messages
                if isinstance(message, dict)
            ]

            results.check(
                roles == ["system", "user", "assistant"],
                f"{expected_path_name} record {position} has incorrect "
                f"message roles: {roles}.",
            )

            assistant_answer = messages[-1].get("content")

            results.check(
                assistant_answer == actual.get("target_item_id"),
                f"{expected_path_name} record {position} assistant "
                "answer does not equal its target item.",
            )

    return record_count


def main() -> None:
    args = parse_arguments()

    if args.max_errors <= 0:
        raise ValueError("--max-errors must be greater than zero.")

    results = ValidationResults(max_errors=args.max_errors)

    split_directory = resolve_project_path(args.split_dir)
    sft_directory = resolve_project_path(args.sft_dir)
    metadata_path = resolve_project_path(args.metadata_file)

    train_sequences_path = (
        split_directory / "train_sequences.jsonl"
    )
    validation_split_path = split_directory / "validation.jsonl"
    test_split_path = split_directory / "test.jsonl"
    catalogue_path = split_directory / "item_catalogue.json"

    train_sft_path = sft_directory / "train.jsonl"
    validation_sft_path = sft_directory / "validation.jsonl"
    sft_summary_path = sft_directory / "sft_summary.json"
    report_path = sft_directory / "validation_report.json"

    required_paths = [
        train_sequences_path,
        validation_split_path,
        test_split_path,
        catalogue_path,
        train_sft_path,
        validation_sft_path,
        sft_summary_path,
        metadata_path,
    ]

    missing_paths = [
        path for path in required_paths if not path.exists()
    ]

    if missing_paths:
        formatted = "\n".join(
            f"  - {path}" for path in missing_paths
        )
        raise FileNotFoundError(
            f"Required validation inputs are missing:\n{formatted}"
        )

    print("=" * 70)
    print("VALIDATING QLORA RECOMMENDER DATASETS")
    print("=" * 70)
    print(f"Dataset:          {args.dataset}")
    print(f"Split directory:  {split_directory}")
    print(f"SFT directory:    {sft_directory}")
    print(f"Metadata file:    {metadata_path}")
    print()

    sft_summary = read_json(sft_summary_path)
    max_history_length = int(
        sft_summary["maximum_history_length"]
    )

    print(f"Maximum history length: {max_history_length}")
    print("Loading catalogue and temporal split records...")

    catalogue_item_ids = load_catalogue(
        catalogue_path=catalogue_path,
        results=results,
    )

    metadata_lookup = load_metadata_lookup(metadata_path)

    train_records = load_unique_user_records(
        path=train_sequences_path,
        split_name="training split",
        results=results,
    )

    validation_records = load_unique_user_records(
        path=validation_split_path,
        split_name="validation split",
        results=results,
    )

    test_records = load_unique_user_records(
        path=test_split_path,
        split_name="test split",
        results=results,
    )

    print("Validating temporal relationships...")

    split_statistics = validate_temporal_splits(
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
        catalogue_item_ids=catalogue_item_ids,
        results=results,
    )

    missing_title_items = [
        item_id
        for item_id in sorted(catalogue_item_ids)
        if not has_usable_title(metadata_lookup.get(item_id))
    ]

    if missing_title_items:
        if args.allow_missing_metadata:
            results.warning(
                f"{len(missing_title_items)} catalogue items have no "
                "usable title and will use the explicit fallback."
            )
        else:
            results.check(
                False,
                f"{len(missing_title_items)} catalogue items have no "
                "usable title: {missing_title_items[:20]}"
            )

    print("Validating training SFT examples...")

    actual_training_examples = compare_sft_records(
        expected_path_name="Training SFT dataset",
        expected_records=expected_training_examples(
            train_records=train_records,
            metadata_lookup=metadata_lookup,
            max_history_length=max_history_length,
        ),
        actual_path=train_sft_path,
        results=results,
    )

    print("Validating validation SFT examples...")

    actual_validation_examples = compare_sft_records(
        expected_path_name="Validation SFT dataset",
        expected_records=expected_validation_examples(
            validation_records=validation_records,
            metadata_lookup=metadata_lookup,
            max_history_length=max_history_length,
        ),
        actual_path=validation_sft_path,
        results=results,
    )

    results.check(
        actual_training_examples
        == split_statistics["expected_training_examples"],
        "Actual training example count does not match the expected count.",
    )

    results.check(
        actual_validation_examples
        == split_statistics["validation_examples"],
        "Actual validation example count does not match the expected count.",
    )

    report = {
        "dataset": args.dataset,
        "status": "passed" if results.passed else "failed",
        "checks_performed": results.checks_performed,
        "total_errors": results.total_errors,
        "stored_error_count": len(results.errors),
        "errors": results.errors,
        "warnings": results.warnings,
        "maximum_history_length": max_history_length,
        "catalogue_items": len(catalogue_item_ids),
        "metadata_records": len(metadata_lookup),
        "missing_title_items": missing_title_items,
        "users": split_statistics["users"],
        "train_interactions": (
            split_statistics["train_interactions"]
        ),
        "expected_training_examples": (
            split_statistics["expected_training_examples"]
        ),
        "actual_training_examples": actual_training_examples,
        "expected_validation_examples": (
            split_statistics["validation_examples"]
        ),
        "actual_validation_examples": actual_validation_examples,
        "test_examples": split_statistics["test_examples"],
    }

    write_json(report, report_path)

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Status:                       {report['status'].upper()}")
    print(f"Checks performed:             {results.checks_performed:,}")
    print(f"Errors:                       {results.total_errors:,}")
    print(f"Warnings:                     {len(results.warnings):,}")
    print(f"Users:                        {report['users']:,}")
    print(f"Catalogue items:              {report['catalogue_items']:,}")
    print(f"Training interactions:        {report['train_interactions']:,}")
    print(f"Training examples:            {actual_training_examples:,}")
    print(f"Validation examples:          {actual_validation_examples:,}")
    print(f"Test examples:                {report['test_examples']:,}")
    print(f"Items without titles:         {len(missing_title_items):,}")
    print(f"Report:                       {report_path}")

    if results.warnings:
        print()
        print("Warnings:")

        for warning in results.warnings:
            print(f"  - {warning}")

    if not results.passed:
        print()
        print("First validation errors:")

        for error in results.errors[:20]:
            print(f"  - {error}")

        raise SystemExit(1)

    print()
    print("The temporal splits and SFT datasets are valid.")


if __name__ == "__main__":
    main()