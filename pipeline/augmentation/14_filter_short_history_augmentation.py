#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from collections import Counter


SYNTHETIC_SOURCE = "llm_synthetic"
EXPECTED_SYNTHETIC_COUNT = 3


def load_json(path: Path):
    print(f"Loading: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_synthetic(history):
    return sum(
        1
        for interaction in history
        if interaction.get("interaction_source") == SYNTHETIC_SOURCE
    )


def get_real_interactions(history):
    return [
        interaction
        for interaction in history
        if interaction.get("interaction_source") != SYNTHETIC_SOURCE
    ]


def interaction_key(interaction):
    """
    Identifies an interaction using the fields that distinguish records
    in the Video Games history files.
    """
    return (
        interaction.get("asin"),
        interaction.get("timestamp"),
        interaction.get("interaction_source"),
        interaction.get("sequence_position"),
        interaction.get("synthetic_position"),
    )


def validate_source_dataset(
    original,
    augmented,
    dataset_name,
):
    """
    Validate that the augmented dataset has the same users as the
    original dataset and exactly three synthetic interactions per user.
    """
    print(f"\nValidating source dataset: {dataset_name}")

    original_users = set(original)
    augmented_users = set(augmented)

    missing_users = original_users - augmented_users
    extra_users = augmented_users - original_users

    if missing_users:
        raise ValueError(
            f"{dataset_name}: {len(missing_users):,} users are missing "
            "from the augmented dataset."
        )

    if extra_users:
        raise ValueError(
            f"{dataset_name}: {len(extra_users):,} unexpected users "
            "found in the augmented dataset."
        )

    synthetic_counts = Counter()

    for user_id, history in augmented.items():
        synthetic_count = count_synthetic(history)
        synthetic_counts[synthetic_count] += 1

        if synthetic_count != EXPECTED_SYNTHETIC_COUNT:
            raise ValueError(
                f"{dataset_name}: user {user_id} has "
                f"{synthetic_count} synthetic interactions; "
                f"expected {EXPECTED_SYNTHETIC_COUNT}."
            )

    print("  Users:", f"{len(augmented):,}")
    print("  Synthetic interactions per user:")
    for count in sorted(synthetic_counts):
        print(
            f"    {count}: {synthetic_counts[count]:,} users"
        )

    print("  Source validation: PASSED")


def validate_filtered_dataset(
    original,
    source_augmented,
    filtered,
    threshold,
    dataset_name,
):
    """
    Validate that:
      - all original users remain
      - all original interactions remain
      - users <= threshold retain 3 synthetic interactions
      - users > threshold retain 0 synthetic interactions
      - validation/test real interactions have not changed
    """
    print(f"\nValidating filtered dataset: {dataset_name}")

    original_users = set(original)
    source_users = set(source_augmented)
    filtered_users = set(filtered)

    if filtered_users != original_users:
        missing = original_users - filtered_users
        extra = filtered_users - original_users

        raise ValueError(
            f"{dataset_name}: user mismatch after filtering. "
            f"Missing={len(missing):,}, Extra={len(extra):,}"
        )

    if source_users != filtered_users:
        raise ValueError(
            f"{dataset_name}: source and filtered user sets differ."
        )

    users_at_or_below_threshold = 0
    users_above_threshold = 0

    synthetic_kept = 0
    synthetic_removed = 0

    errors = []

    for user_id in original:
        original_history = original[user_id]
        source_history = source_augmented[user_id]
        filtered_history = filtered[user_id]

        original_length = len(original_history)
        source_synthetic_count = count_synthetic(source_history)
        filtered_synthetic_count = count_synthetic(filtered_history)

        # ------------------------------------------------------------
        # Check expected synthetic count
        # ------------------------------------------------------------

        if original_length <= threshold:
            users_at_or_below_threshold += 1

            if filtered_synthetic_count != EXPECTED_SYNTHETIC_COUNT:
                errors.append(
                    f"{user_id}: expected 3 synthetic interactions "
                    f"but found {filtered_synthetic_count}."
                )

        else:
            users_above_threshold += 1

            if filtered_synthetic_count != 0:
                errors.append(
                    f"{user_id}: expected 0 synthetic interactions "
                    f"but found {filtered_synthetic_count}."
                )

        synthetic_kept += filtered_synthetic_count
        synthetic_removed += (
            source_synthetic_count - filtered_synthetic_count
        )

        # ------------------------------------------------------------
        # Check that every original interaction remains
        # ------------------------------------------------------------

        original_keys = Counter(
            interaction_key(interaction)
            for interaction in original_history
        )

        filtered_real_history = get_real_interactions(filtered_history)

        filtered_real_keys = Counter(
            interaction_key(interaction)
            for interaction in filtered_real_history
        )

        for key, count in original_keys.items():
            if filtered_real_keys[key] < count:
                errors.append(
                    f"{user_id}: original interaction missing "
                    f"after filtering: {key}"
                )

        # ------------------------------------------------------------
        # Check validation/test preservation
        #
        # The final two real interactions in the original sequence
        # should remain exactly the same in the filtered dataset.
        # ------------------------------------------------------------

        original_real_history = get_real_interactions(original_history)

        if len(original_real_history) >= 2:
            original_validation = original_real_history[-2]
            original_test = original_real_history[-1]

            filtered_real_history = get_real_interactions(
                filtered_history
            )

            if len(filtered_real_history) < 2:
                errors.append(
                    f"{user_id}: fewer than two real interactions "
                    "remain after filtering."
                )
            else:
                filtered_validation = filtered_real_history[-2]
                filtered_test = filtered_real_history[-1]

                if filtered_validation != original_validation:
                    errors.append(
                        f"{user_id}: validation interaction changed."
                    )

                if filtered_test != original_test:
                    errors.append(
                        f"{user_id}: test interaction changed."
                    )

    if errors:
        print("\nValidation errors:")
        for error in errors[:20]:
            print("  -", error)

        if len(errors) > 20:
            print(
                f"  ... and {len(errors) - 20:,} more errors."
            )

        raise ValueError(
            f"{dataset_name}: validation FAILED with "
            f"{len(errors):,} error(s)."
        )

    print("  Users:", f"{len(filtered):,}")
    print(
        "  Users <= threshold:",
        f"{users_at_or_below_threshold:,}",
    )
    print(
        "  Users > threshold:",
        f"{users_above_threshold:,}",
    )
    print(
        "  Synthetic interactions kept:",
        f"{synthetic_kept:,}",
    )
    print(
        "  Synthetic interactions removed:",
        f"{synthetic_removed:,}",
    )
    print("  Filtered validation: PASSED")


def filter_dataset(
    original,
    augmented,
    threshold,
):
    """
    Create the selective augmentation dataset.

    Users with original history length <= threshold retain all
    three synthetic interactions.

    Users with original history length > threshold have only
    their synthetic interactions removed.

    All original interactions are preserved.
    """
    filtered = {}

    for user_id, augmented_history in augmented.items():
        original_length = len(original[user_id])

        if original_length <= threshold:
            # Short-history user:
            # Keep the complete augmented sequence.
            filtered[user_id] = augmented_history.copy()

        else:
            # Longer-history user:
            # Remove ONLY synthetic LLM interactions.
            filtered[user_id] = [
                interaction
                for interaction in augmented_history
                if interaction.get("interaction_source")
                != SYNTHETIC_SOURCE
            ]

    return filtered


def save_json(data, path: Path):
    print(f"\nSaving: {path}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("  Saved successfully.")


def process_model(
    model_name,
    original,
    augmented_path,
    output_path,
    threshold,
):
    print("\n" + "=" * 70)
    print(f"PROCESSING {model_name.upper()}")
    print("=" * 70)

    augmented = load_json(augmented_path)

    # Validate the original full augmentation before modifying it.
    validate_source_dataset(
        original=original,
        augmented=augmented,
        dataset_name=f"{model_name} full augmentation",
    )

    # Create selective augmentation.
    filtered = filter_dataset(
        original=original,
        augmented=augmented,
        threshold=threshold,
    )

    # Validate the new dataset.
    validate_filtered_dataset(
        original=original,
        source_augmented=augmented,
        filtered=filtered,
        threshold=threshold,
        dataset_name=f"{model_name} selective augmentation",
    )

    # Save only after validation passes.
    save_json(
        filtered,
        output_path,
    )

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create selective Video Games LLM augmentation datasets "
            "by retaining the three synthetic interactions only for "
            "users whose original history length is at or below "
            "the specified threshold."
        )
    )

    parser.add_argument(
        "--original",
        required=True,
        help="Path to the original Video Games user histories JSON.",
    )

    parser.add_argument(
        "--qwen-input",
        required=True,
        help="Path to the full Qwen augmented user histories JSON.",
    )

    parser.add_argument(
        "--llama-input",
        required=True,
        help="Path to the full LLaMA augmented user histories JSON.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the two selective JSON files will be saved.",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=7,
        help=(
            "Maximum original history length for retaining the "
            "three synthetic interactions. Default: 7."
        ),
    )

    args = parser.parse_args()

    if args.threshold < 0:
        raise ValueError("Threshold must be >= 0.")

    original_path = Path(args.original)
    qwen_input_path = Path(args.qwen_input)
    llama_input_path = Path(args.llama_input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("VIDEO GAMES — SELECTIVE LLM AUGMENTATION")
    print("=" * 70)
    print(f"Threshold: original history <= {args.threshold}")
    print()

    # ------------------------------------------------------------
    # Load original dataset once.
    # ------------------------------------------------------------

    original = load_json(original_path)

    print(
        f"\nOriginal users loaded: {len(original):,}"
    )

    # ------------------------------------------------------------
    # Process Qwen.
    # ------------------------------------------------------------

    qwen_output_path = (
        output_dir
        / "Video_Games_selective_qwen_augmented_user_histories.json"
    )

    process_model(
        model_name="Qwen",
        original=original,
        augmented_path=qwen_input_path,
        output_path=qwen_output_path,
        threshold=args.threshold,
    )

    # ------------------------------------------------------------
    # Process LLaMA.
    # ------------------------------------------------------------

    llama_output_path = (
        output_dir
        / "Video_Games_selective_llama_augmented_user_histories.json"
    )

    process_model(
        model_name="LLaMA",
        original=original,
        augmented_path=llama_input_path,
        output_path=llama_output_path,
        threshold=args.threshold,
    )

    # ------------------------------------------------------------
    # Final summary.
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SELECTIVE AUGMENTATION COMPLETE")
    print("=" * 70)

    print("\nCreated files:")

    print(f"  Qwen:")
    print(f"    {qwen_output_path}")

    print(f"\n  LLaMA:")
    print(f"    {llama_output_path}")

    print(
        "\nFiltering rule:",
        f"users with original history <= {args.threshold} "
        "retain 3 synthetic interactions;",
    )
    print(
        f"users with original history > {args.threshold} "
        "retain 0 synthetic interactions."
    )

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()