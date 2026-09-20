"""
Build supervised fine-tuning datasets for the QLoRA recommender.

Training:
    Creates multiple prefix -> next-item examples from each user's
    training sequence. Optionally retains only the most recent N
    examples per user.

Validation:
    Creates one history -> held-out next-item example per user.

The history shown to the LLM contains both:
    - Exact item ID/ASIN
    - Product title

The expected assistant response is always the exact item ID.
"""

from __future__ import annotations

import argparse
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
    write_jsonl,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build QLoRA SFT datasets from temporal splits."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, for example Fashion or Video Games.",
    )

    parser.add_argument(
        "--split-dir",
        required=True,
        help="Directory containing the Step 01 output files.",
    )

    parser.add_argument(
        "--metadata-file",
        required=True,
        help="Item metadata lookup containing product titles.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to an sft folder alongside "
            "the splits folder."
        ),
    )

    parser.add_argument(
        "--max-history-length",
        type=int,
        default=50,
        help="Maximum number of recent items included in a prompt.",
    )

    parser.add_argument(
        "--max-training-examples-per-user",
        type=int,
        default=None,
        help=(
            "Optional cap on training examples contributed by each "
            "user. When set, the most recent examples are retained. "
            "By default, every eligible prefix-to-next-item example "
            "is written."
        ),
    )

    parser.add_argument(
        "--train-output-name",
        default="train.jsonl",
        help="Training output filename within the output directory.",
    )

    parser.add_argument(
        "--validation-output-name",
        default="validation.jsonl",
        help="Validation output filename within the output directory.",
    )

    parser.add_argument(
        "--summary-output-name",
        default="sft_summary.json",
        help="Summary output filename within the output directory.",
    )

    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help=(
            "Allow items without titles. By default, missing titles "
            "cause the script to stop."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing SFT files to be replaced.",
    )

    return parser.parse_args()


def determine_output_directory(
    split_directory: Path,
    output_dir: str | None,
) -> Path:
    if output_dir is not None:
        return resolve_project_path(output_dir)

    return split_directory.parent / "sft"


def has_usable_title(metadata: dict[str, Any] | None) -> bool:
    if not metadata:
        return False

    for field in ("title", "name", "product_title"):
        if clean_text(metadata.get(field)):
            return True

    return False


def load_catalogue_item_ids(catalogue_path: Path) -> set[str]:
    catalogue = read_json(catalogue_path)

    if not isinstance(catalogue, dict):
        raise ValueError("Item catalogue must be a JSON object.")

    items = catalogue.get("items")

    if not isinstance(items, list):
        raise ValueError(
            "Item catalogue must contain an 'items' list."
        )

    item_ids: set[str] = set()

    for index, item in enumerate(items):
        if not isinstance(item, dict) or not item.get("item_id"):
            raise ValueError(
                f"Invalid catalogue record at position {index}."
            )

        item_ids.add(str(item["item_id"]))

    return item_ids


def check_metadata_coverage(
    catalogue_item_ids: set[str],
    metadata_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    missing_items = []

    for item_id in sorted(catalogue_item_ids):
        metadata = metadata_lookup.get(item_id)

        if not has_usable_title(metadata):
            missing_items.append(item_id)

    return missing_items


def build_messages_with_answer(
    history_item_ids: list[str],
    target_item_id: str,
    metadata_lookup: dict[str, dict[str, Any]],
    max_history_length: int,
) -> list[dict[str, str]]:
    messages = build_recommendation_messages(
        history_item_ids=history_item_ids,
        metadata_lookup=metadata_lookup,
        max_history_length=max_history_length,
    )

    messages.append(
        {
            "role": "assistant",
            "content": str(target_item_id),
        }
    )

    return messages


def iter_training_examples(
    train_path: Path,
    metadata_lookup: dict[str, dict[str, Any]],
    max_history_length: int,
    max_training_examples_per_user: int | None,
    statistics: dict[str, int],
) -> Iterator[dict[str, Any]]:
    """
    Expand each training sequence into prefix-to-next-item examples.

    Example:
        [A, B, C]

    Becomes:
        [A]    -> B
        [A, B] -> C
    """

    for record_number, record in enumerate(
        iter_jsonl(train_path),
        start=1,
    ):
        require_fields(
            record,
            ["user_id", "item_ids", "timestamps"],
            context=f"training record {record_number}",
        )

        user_id = str(record["user_id"])
        item_ids = [str(item_id) for item_id in record["item_ids"]]
        timestamps = record["timestamps"]

        if len(item_ids) != len(timestamps):
            raise ValueError(
                f"Item and timestamp counts differ for user {user_id}."
            )

        if len(item_ids) < 2:
            raise ValueError(
                f"Training sequence for user {user_id} contains fewer "
                "than two items."
            )

        statistics["training_users"] += 1

        available_example_count = len(item_ids) - 1
        statistics["available_training_examples"] += (
            available_example_count
        )

        first_target_position = 1

        if max_training_examples_per_user is not None:
            retained_example_count = min(
                available_example_count,
                max_training_examples_per_user,
            )
            first_target_position = (
                len(item_ids) - retained_example_count
            )

            omitted_example_count = (
                available_example_count - retained_example_count
            )
            statistics["omitted_training_examples"] += (
                omitted_example_count
            )

            if omitted_example_count:
                statistics["users_capped"] += 1

        for target_position in range(
            first_target_position,
            len(item_ids),
        ):
            full_history = item_ids[:target_position]
            full_history_timestamps = timestamps[:target_position]

            model_history = full_history[-max_history_length:]
            model_history_timestamps = (
                full_history_timestamps[-max_history_length:]
            )

            target_item_id = item_ids[target_position]
            target_timestamp = timestamps[target_position]

            example = {
                "example_id": (
                    f"{user_id}:train:{target_position}"
                ),
                "dataset": statistics["dataset"],
                "split": "train",
                "user_id": user_id,
                "history_item_ids": model_history,
                "history_timestamps": model_history_timestamps,
                "target_item_id": target_item_id,
                "target_timestamp": target_timestamp,
                "history_length": len(model_history),
                "original_history_length": len(full_history),
                "history_truncated": (
                    len(full_history) > max_history_length
                ),
                "messages": build_messages_with_answer(
                    history_item_ids=model_history,
                    target_item_id=target_item_id,
                    metadata_lookup=metadata_lookup,
                    max_history_length=max_history_length,
                ),
            }

            statistics["training_examples"] += 1

            if example["history_truncated"]:
                statistics["truncated_training_examples"] += 1

            yield example


def iter_validation_examples(
    validation_path: Path,
    metadata_lookup: dict[str, dict[str, Any]],
    max_history_length: int,
    statistics: dict[str, int],
) -> Iterator[dict[str, Any]]:
    """Create one validation SFT example per eligible user."""

    for record_number, record in enumerate(
        iter_jsonl(validation_path),
        start=1,
    ):
        require_fields(
            record,
            [
                "user_id",
                "history_item_ids",
                "history_timestamps",
                "target_item_id",
                "target_timestamp",
            ],
            context=f"validation record {record_number}",
        )

        user_id = str(record["user_id"])
        full_history = [
            str(item_id)
            for item_id in record["history_item_ids"]
        ]
        full_history_timestamps = record["history_timestamps"]
        target_item_id = str(record["target_item_id"])

        if len(full_history) != len(full_history_timestamps):
            raise ValueError(
                f"Validation item and timestamp counts differ for "
                f"user {user_id}."
            )

        if not full_history:
            raise ValueError(
                f"Validation history is empty for user {user_id}."
            )

        model_history = full_history[-max_history_length:]
        model_history_timestamps = (
            full_history_timestamps[-max_history_length:]
        )

        example = {
            "example_id": f"{user_id}:validation",
            "dataset": statistics["dataset"],
            "split": "validation",
            "user_id": user_id,
            "history_item_ids": model_history,
            "history_timestamps": model_history_timestamps,
            "target_item_id": target_item_id,
            "target_timestamp": record["target_timestamp"],
            "history_length": len(model_history),
            "original_history_length": len(full_history),
            "history_truncated": (
                len(full_history) > max_history_length
            ),
            "messages": build_messages_with_answer(
                history_item_ids=model_history,
                target_item_id=target_item_id,
                metadata_lookup=metadata_lookup,
                max_history_length=max_history_length,
            ),
        }

        statistics["validation_examples"] += 1

        if example["history_truncated"]:
            statistics["truncated_validation_examples"] += 1

        yield example


def check_existing_outputs(
    output_paths: list[Path],
    overwrite: bool,
) -> None:
    existing = [
        path for path in output_paths if path.exists()
    ]

    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)

        raise FileExistsError(
            "SFT output files already exist:\n"
            f"{formatted}\n"
            "Use --overwrite only if you intentionally want to "
            "rebuild them."
        )


def validate_output_filename(
    filename: str,
    expected_suffix: str,
    argument_name: str,
) -> str:
    """Validate that an output name is a plain filename of the right type."""

    candidate = Path(filename)

    if (
        not filename
        or candidate.name != filename
        or filename in {".", ".."}
    ):
        raise ValueError(
            f"{argument_name} must be a filename without directories."
        )

    if candidate.suffix.lower() != expected_suffix:
        raise ValueError(
            f"{argument_name} must end with {expected_suffix}."
        )

    return filename


def main() -> None:
    args = parse_arguments()

    if args.max_history_length <= 0:
        raise ValueError(
            "--max-history-length must be greater than zero."
        )

    if (
        args.max_training_examples_per_user is not None
        and args.max_training_examples_per_user <= 0
    ):
        raise ValueError(
            "--max-training-examples-per-user must be greater than "
            "zero when provided."
        )

    train_output_name = validate_output_filename(
        args.train_output_name,
        expected_suffix=".jsonl",
        argument_name="--train-output-name",
    )
    validation_output_name = validate_output_filename(
        args.validation_output_name,
        expected_suffix=".jsonl",
        argument_name="--validation-output-name",
    )
    summary_output_name = validate_output_filename(
        args.summary_output_name,
        expected_suffix=".json",
        argument_name="--summary-output-name",
    )

    split_directory = resolve_project_path(args.split_dir)
    metadata_path = resolve_project_path(args.metadata_file)

    train_path = split_directory / "train_sequences.jsonl"
    validation_path = split_directory / "validation.jsonl"
    catalogue_path = split_directory / "item_catalogue.json"

    required_inputs = [
        train_path,
        validation_path,
        catalogue_path,
        metadata_path,
    ]

    missing_inputs = [
        path for path in required_inputs if not path.exists()
    ]

    if missing_inputs:
        formatted = "\n".join(
            f"  - {path}" for path in missing_inputs
        )

        raise FileNotFoundError(
            f"Required Step 02 inputs are missing:\n{formatted}"
        )

    output_directory = determine_output_directory(
        split_directory=split_directory,
        output_dir=args.output_dir,
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    train_output = output_directory / train_output_name
    validation_output = output_directory / validation_output_name
    summary_output = output_directory / summary_output_name

    check_existing_outputs(
        [train_output, validation_output, summary_output],
        overwrite=args.overwrite,
    )

    print("=" * 70)
    print("BUILDING QLORA SFT DATASETS")
    print("=" * 70)
    print(f"Dataset:            {args.dataset}")
    print(f"Split directory:    {split_directory}")
    print(f"Metadata file:      {metadata_path}")
    print(f"Output directory:   {output_directory}")
    print(f"Max history length: {args.max_history_length}")
    print(
        "Max train examples/user: "
        f"{args.max_training_examples_per_user or 'unlimited'}"
    )
    print(f"Training output:    {train_output.name}")
    print(f"Validation output:  {validation_output.name}")
    print(f"Summary output:     {summary_output.name}")
    print()

    print("Loading metadata and checking catalogue coverage...")

    metadata_lookup = load_metadata_lookup(metadata_path)
    catalogue_item_ids = load_catalogue_item_ids(catalogue_path)

    missing_metadata = check_metadata_coverage(
        catalogue_item_ids=catalogue_item_ids,
        metadata_lookup=metadata_lookup,
    )

    metadata_coverage = (
        (len(catalogue_item_ids) - len(missing_metadata))
        / len(catalogue_item_ids)
    )

    print(f"Catalogue items:       {len(catalogue_item_ids):,}")
    print(f"Metadata records:      {len(metadata_lookup):,}")
    print(f"Items with titles:     "
          f"{len(catalogue_item_ids) - len(missing_metadata):,}")
    print(f"Items missing titles:  {len(missing_metadata):,}")
    print(f"Metadata coverage:     {metadata_coverage:.2%}")
    print()

    if missing_metadata and not args.allow_missing_metadata:
        preview = "\n".join(
            f"  - {item_id}"
            for item_id in missing_metadata[:20]
        )

        raise ValueError(
            f"{len(missing_metadata):,} catalogue items do not have "
            "usable product titles.\n"
            f"First missing IDs:\n{preview}\n"
            "Fix the metadata lookup or use --allow-missing-metadata "
            "only if missing titles are intentional."
        )

    statistics: dict[str, Any] = {
        "dataset": args.dataset,
        "training_users": 0,
        "available_training_examples": 0,
        "training_examples": 0,
        "omitted_training_examples": 0,
        "users_capped": 0,
        "validation_examples": 0,
        "truncated_training_examples": 0,
        "truncated_validation_examples": 0,
    }

    print("Building training examples...")

    write_jsonl(
        iter_training_examples(
            train_path=train_path,
            metadata_lookup=metadata_lookup,
            max_history_length=args.max_history_length,
            max_training_examples_per_user=(
                args.max_training_examples_per_user
            ),
            statistics=statistics,
        ),
        train_output,
    )

    print("Building validation examples...")

    write_jsonl(
        iter_validation_examples(
            validation_path=validation_path,
            metadata_lookup=metadata_lookup,
            max_history_length=args.max_history_length,
            statistics=statistics,
        ),
        validation_output,
    )

    summary = {
        "dataset": args.dataset,
        "format": "chat_supervised_fine_tuning",
        "target_format": "exact_item_id",
        "maximum_history_length": args.max_history_length,
        "maximum_training_examples_per_user": (
            args.max_training_examples_per_user
        ),
        "catalogue_items": len(catalogue_item_ids),
        "metadata_records": len(metadata_lookup),
        "items_with_titles": (
            len(catalogue_item_ids) - len(missing_metadata)
        ),
        "items_missing_titles": len(missing_metadata),
        "metadata_coverage": metadata_coverage,
        "training_users": statistics["training_users"],
        "available_training_examples": (
            statistics["available_training_examples"]
        ),
        "training_examples": statistics["training_examples"],
        "omitted_training_examples": (
            statistics["omitted_training_examples"]
        ),
        "users_capped": statistics["users_capped"],
        "validation_examples": statistics["validation_examples"],
        "truncated_training_examples": (
            statistics["truncated_training_examples"]
        ),
        "truncated_validation_examples": (
            statistics["truncated_validation_examples"]
        ),
        "files": {
            "train": str(train_output),
            "validation": str(validation_output),
            "metadata": str(metadata_path),
            "catalogue": str(catalogue_path),
        },
    }

    write_json(summary, summary_output)

    print()
    print("=" * 70)
    print("SFT DATASET CREATION COMPLETE")
    print("=" * 70)
    print(
        f"Training users:                "
        f"{statistics['training_users']:,}"
    )
    print(
        f"Available training examples:   "
        f"{statistics['available_training_examples']:,}"
    )
    print(
        f"Training examples:             "
        f"{statistics['training_examples']:,}"
    )
    print(
        f"Omitted training examples:     "
        f"{statistics['omitted_training_examples']:,}"
    )
    print(
        f"Users affected by cap:         "
        f"{statistics['users_capped']:,}"
    )
    print(
        f"Validation examples:           "
        f"{statistics['validation_examples']:,}"
    )
    print(
        f"Truncated training examples:   "
        f"{statistics['truncated_training_examples']:,}"
    )
    print(
        f"Truncated validation examples: "
        f"{statistics['truncated_validation_examples']:,}"
    )
    print()
    print(f"Training file:   {train_output}")
    print(f"Validation file: {validation_output}")
    print(f"Summary file:    {summary_output}")


if __name__ == "__main__":
    main()
