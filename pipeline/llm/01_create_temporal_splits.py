"""
Create chronological train, validation, and test splits for
LLM-based sequential recommendation.

For a user sequence:

    [i1, i2, ..., i(n-2), i(n-1), i(n)]

the script creates:

    Train sequence:
        [i1, ..., i(n-2)]

    Validation:
        history = [i1, ..., i(n-2)]
        target  = i(n-1)

    Test:
        history = [i1, ..., i(n-1)]
        target  = i(n)

The real validation item is included in the test history.
No model-generated item is ever inserted into a history.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    iter_recbole_interactions,
    resolve_project_path,
    write_json,
    write_jsonl,
)


OUTPUT_FILENAMES = {
    "train": "train_sequences.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
    "catalogue": "item_catalogue.json",
    "summary": "split_summary.json",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create chronological train, validation, and test splits "
            "from a RecBole .inter file."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name used in the saved summary, for example Fashion.",
    )

    parser.add_argument(
        "--inter-file",
        required=True,
        help=(
            "Path to the original filtered RecBole .inter file, relative "
            "to the project root or absolute."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. By default, outputs are written to "
            "<inter-file-parent>/llm_recommender/splits/."
        ),
    )

    parser.add_argument(
        "--min-interactions",
        type=int,
        default=4,
        help=(
            "Minimum interactions required per user. Four interactions "
            "allow at least one trainable next-item example plus validation "
            "and test targets. Default: 4."
        ),
    )

    parser.add_argument(
        "--user-field",
        default="user_id",
        help="User ID field in the .inter file. Default: user_id.",
    )

    parser.add_argument(
        "--item-field",
        default="item_id",
        help="Item ID field in the .inter file. Default: item_id.",
    )

    parser.add_argument(
        "--time-field",
        default="timestamp",
        help="Timestamp field in the .inter file. Default: timestamp.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing split files to be replaced.",
    )

    return parser.parse_args()


def calculate_sha256(path: Path) -> str:
    """Calculate a reproducibility fingerprint for the input file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def determine_output_directory(
    inter_path: Path,
    output_dir: str | None,
) -> Path:
    """Resolve the requested or default output directory."""

    if output_dir is not None:
        return resolve_project_path(output_dir)

    return inter_path.parent / "llm_recommender" / "splits"


def check_output_files(
    output_directory: Path,
    overwrite: bool,
) -> None:
    """
    Prevent accidental replacement of an existing experimental split.
    """

    existing_files = [
        output_directory / filename
        for filename in OUTPUT_FILENAMES.values()
        if (output_directory / filename).exists()
    ]

    if existing_files and not overwrite:
        formatted_paths = "\n".join(
            f"  - {path}" for path in existing_files
        )

        raise FileExistsError(
            "Split output files already exist:\n"
            f"{formatted_paths}\n"
            "Use --overwrite only if you intentionally want to recreate "
            "the split."
        )


def load_user_interactions(
    inter_path: Path,
    user_field: str,
    item_field: str,
    time_field: str,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    Counter[str],
    int,
]:
    """
    Read the .inter file and group records by user.
    """

    user_interactions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    item_counts: Counter[str] = Counter()
    interaction_count = 0

    for interaction in iter_recbole_interactions(
        path=inter_path,
        user_field=user_field,
        item_field=item_field,
        time_field=time_field,
    ):
        user_id = interaction["user_id"]
        item_id = interaction["item_id"]

        user_interactions[user_id].append(interaction)
        item_counts[item_id] += 1
        interaction_count += 1

    return dict(user_interactions), item_counts, interaction_count


def create_user_splits(
    user_id: str,
    interactions: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """
    Create the three temporal records for one eligible user.
    """

    ordered = sorted(
        interactions,
        key=lambda record: (
            record["timestamp"],
            record["row_number"],
        ),
    )

    item_ids = [record["item_id"] for record in ordered]
    timestamps = [record["timestamp"] for record in ordered]

    train_item_ids = item_ids[:-2]
    train_timestamps = timestamps[:-2]

    validation_target = item_ids[-2]
    validation_target_timestamp = timestamps[-2]

    test_target = item_ids[-1]
    test_target_timestamp = timestamps[-1]

    validation_history = list(train_item_ids)
    validation_history_timestamps = list(train_timestamps)

    test_history = item_ids[:-1]
    test_history_timestamps = timestamps[:-1]

    train_record = {
        "user_id": user_id,
        "item_ids": train_item_ids,
        "timestamps": train_timestamps,
        "sequence_length": len(train_item_ids),
    }

    validation_record = {
        "user_id": user_id,
        "history_item_ids": validation_history,
        "history_timestamps": validation_history_timestamps,
        "target_item_id": validation_target,
        "target_timestamp": validation_target_timestamp,
        "history_length": len(validation_history),
    }

    test_record = {
        "user_id": user_id,
        "history_item_ids": test_history,
        "history_timestamps": test_history_timestamps,
        "target_item_id": test_target,
        "target_timestamp": test_target_timestamp,
        "history_length": len(test_history),
    }

    validate_user_split(
        train_record=train_record,
        validation_record=validation_record,
        test_record=test_record,
    )

    return train_record, validation_record, test_record


def validate_user_split(
    train_record: dict[str, Any],
    validation_record: dict[str, Any],
    test_record: dict[str, Any],
) -> None:
    """
    Check the critical temporal relationships for one user.
    """

    user_id = train_record["user_id"]

    if validation_record["user_id"] != user_id:
        raise AssertionError(f"Validation user mismatch for {user_id}.")

    if test_record["user_id"] != user_id:
        raise AssertionError(f"Test user mismatch for {user_id}.")

    train_items = train_record["item_ids"]
    validation_history = validation_record["history_item_ids"]
    test_history = test_record["history_item_ids"]

    if validation_history != train_items:
        raise AssertionError(
            f"Validation history does not equal the training sequence "
            f"for user {user_id}."
        )

    expected_test_history = (
        train_items + [validation_record["target_item_id"]]
    )

    if test_history != expected_test_history:
        raise AssertionError(
            f"Test history does not contain the real validation target "
            f"for user {user_id}."
        )

    if (
        validation_record["target_timestamp"]
        > test_record["target_timestamp"]
    ):
        raise AssertionError(
            f"Validation occurs after testing for user {user_id}."
        )


def build_catalogue(
    item_counts: Counter[str],
) -> dict[str, Any]:
    """
    Build a deterministic catalogue containing every valid dataset item.

    The catalogue includes all items in the filtered dataset, matching
    the full-catalogue evaluation setting.
    """

    items = [
        {
            "item_id": item_id,
            "interaction_count": item_counts[item_id],
        }
        for item_id in sorted(item_counts)
    ]

    return {
        "num_items": len(items),
        "items": items,
    }


def main() -> None:
    args = parse_arguments()

    if args.min_interactions < 4:
        raise ValueError(
            "--min-interactions must be at least 4. A user needs enough "
            "interactions for training, validation, and testing."
        )

    inter_path = resolve_project_path(args.inter_file)

    if not inter_path.exists():
        raise FileNotFoundError(
            f"Interaction file not found: {inter_path}"
        )

    output_directory = determine_output_directory(
        inter_path=inter_path,
        output_dir=args.output_dir,
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    check_output_files(
        output_directory=output_directory,
        overwrite=args.overwrite,
    )

    print("=" * 70)
    print("CREATING LLM RECOMMENDER TEMPORAL SPLITS")
    print("=" * 70)
    print(f"Dataset:       {args.dataset}")
    print(f"Input file:    {inter_path}")
    print(f"Output folder: {output_directory}")
    print(f"Minimum interactions per user: {args.min_interactions}")
    print()

    print("Loading and grouping interactions...")

    user_interactions, item_counts, total_interactions = (
        load_user_interactions(
            inter_path=inter_path,
            user_field=args.user_field,
            item_field=args.item_field,
            time_field=args.time_field,
        )
    )

    print(f"Users loaded:        {len(user_interactions):,}")
    print(f"Interactions loaded: {total_interactions:,}")
    print(f"Unique items:        {len(item_counts):,}")
    print()

    eligible_users = 0
    skipped_users = 0
    train_interaction_count = 0
    expected_train_example_count = 0

    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    # Sorting user IDs makes the output deterministic.
    for user_id in sorted(user_interactions):
        interactions = user_interactions[user_id]

        if len(interactions) < args.min_interactions:
            skipped_users += 1
            continue

        train_record, validation_record, test_record = (
            create_user_splits(
                user_id=user_id,
                interactions=interactions,
            )
        )

        train_records.append(train_record)
        validation_records.append(validation_record)
        test_records.append(test_record)

        eligible_users += 1
        train_interaction_count += train_record["sequence_length"]

        # A training sequence of length m creates m - 1
        # prefix-to-next-item SFT examples.
        expected_train_example_count += (
            train_record["sequence_length"] - 1
        )

    if eligible_users == 0:
        raise ValueError(
            "No users qualified for the temporal split. Check the input "
            "dataset and --min-interactions value."
        )

    if not (
        len(train_records)
        == len(validation_records)
        == len(test_records)
    ):
        raise AssertionError(
            "Train, validation, and test user counts do not match."
        )

    train_path = output_directory / OUTPUT_FILENAMES["train"]
    validation_path = (
        output_directory / OUTPUT_FILENAMES["validation"]
    )
    test_path = output_directory / OUTPUT_FILENAMES["test"]
    catalogue_path = output_directory / OUTPUT_FILENAMES["catalogue"]
    summary_path = output_directory / OUTPUT_FILENAMES["summary"]

    print("Writing split files...")

    write_jsonl(train_records, train_path)
    write_jsonl(validation_records, validation_path)
    write_jsonl(test_records, test_path)

    catalogue = build_catalogue(item_counts)
    write_json(catalogue, catalogue_path)

    summary = {
        "dataset": args.dataset,
        "source_interaction_file": str(inter_path),
        "source_sha256": calculate_sha256(inter_path),
        "splitting_strategy": "chronological_leave_two_out",
        "minimum_interactions": args.min_interactions,
        "total_users_before_filtering": len(user_interactions),
        "eligible_users": eligible_users,
        "skipped_users": skipped_users,
        "total_interactions_before_splitting": total_interactions,
        "train_interactions": train_interaction_count,
        "validation_interactions": eligible_users,
        "test_interactions": eligible_users,
        "expected_training_examples": expected_train_example_count,
        "catalogue_items": len(item_counts),
        "files": {
            "train": str(train_path),
            "validation": str(validation_path),
            "test": str(test_path),
            "catalogue": str(catalogue_path),
        },
    }

    write_json(summary, summary_path)

    print()
    print("=" * 70)
    print("TEMPORAL SPLIT COMPLETE")
    print("=" * 70)
    print(f"Eligible users:             {eligible_users:,}")
    print(f"Skipped users:              {skipped_users:,}")
    print(f"Training interactions:      {train_interaction_count:,}")
    print(f"Expected training examples: {expected_train_example_count:,}")
    print(f"Validation targets:         {eligible_users:,}")
    print(f"Test targets:               {eligible_users:,}")
    print(f"Catalogue items:            {len(item_counts):,}")
    print()
    print(f"Train:      {train_path}")
    print(f"Validation: {validation_path}")
    print(f"Test:       {test_path}")
    print(f"Catalogue:  {catalogue_path}")
    print(f"Summary:    {summary_path}")


if __name__ == "__main__":
    main()