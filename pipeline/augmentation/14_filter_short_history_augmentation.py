#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


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
    Create a key for identifying an interaction.

    We use the fields that distinguish interactions in the
    Video Games user-history files.
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
    Validate the existing full augmented dataset before filtering.

    Checks:
      - same users as original dataset
      - exactly 3 synthetic interactions per user
    """

    print()
    print("=" * 70)
    print(f"VALIDATING SOURCE DATASET: {dataset_name}")
    print("=" * 70)

    original_users = set(original)
    augmented_users = set(augmented)

    missing_users = original_users - augmented_users
    extra_users = augmented_users - original_users

    if missing_users:
        raise ValueError(
            f"{dataset_name}: {len(missing_users):,} users are "
            "missing from the augmented dataset."
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

    print(f"Users: {len(augmented):,}")

    print("\nSynthetic interactions per user:")

    for count in sorted(synthetic_counts):
        print(
            f"  {count}: "
            f"{synthetic_counts[count]:,} users"
        )

    print("\nSource validation: PASSED")


def filter_dataset(
    original,
    augmented,
    threshold,
):
    """
    Create the selective augmentation dataset.

    If a user's ORIGINAL history length is <= threshold:
        keep all 3 synthetic interactions.

    If a user's ORIGINAL history length is > threshold:
        remove ONLY the synthetic interactions.

    All original interactions are preserved.
    """

    filtered = {}

    users_at_or_below_threshold = 0
    users_above_threshold = 0

    synthetic_kept = 0
    synthetic_removed = 0

    for user_id, augmented_history in augmented.items():

        original_length = len(original[user_id])

        if original_length <= threshold:

            # Short-history user:
            # Keep the complete augmented sequence.

            filtered_history = augmented_history.copy()

            users_at_or_below_threshold += 1
            synthetic_kept += count_synthetic(filtered_history)

        else:

            # Longer-history user:
            # Remove ONLY LLM synthetic interactions.

            filtered_history = [
                interaction
                for interaction in augmented_history
                if interaction.get("interaction_source")
                != SYNTHETIC_SOURCE
            ]

            users_above_threshold += 1

            synthetic_removed += (
                count_synthetic(augmented_history)
                - count_synthetic(filtered_history)
            )

        filtered[user_id] = filtered_history

    print()
    print("=" * 70)
    print("FILTERING SUMMARY")
    print("=" * 70)

    print(
        f"Users with original history <= {threshold}: "
        f"{users_at_or_below_threshold:,}"
    )

    print(
        f"Users with original history > {threshold}: "
        f"{users_above_threshold:,}"
    )

    print(
        f"Synthetic interactions kept: "
        f"{synthetic_kept:,}"
    )

    print(
        f"Synthetic interactions removed: "
        f"{synthetic_removed:,}"
    )

    return filtered


def validate_filtered_dataset(
    original,
    source_augmented,
    filtered,
    threshold,
    dataset_name,
):
    """
    Thoroughly validate the selective dataset.

    Checks:
      1. Same users as original dataset.
      2. Same users as source augmented dataset.
      3. Users <= threshold retain exactly 3 synthetic interactions.
      4. Users > threshold retain zero synthetic interactions.
      5. Every original interaction remains.
      6. Original validation interaction remains unchanged.
      7. Original test interaction remains unchanged.
    """

    print()
    print("=" * 70)
    print(f"VALIDATING FILTERED DATASET: {dataset_name}")
    print("=" * 70)

    original_users = set(original)
    source_users = set(source_augmented)
    filtered_users = set(filtered)

    # ------------------------------------------------------------
    # User-set validation
    # ------------------------------------------------------------

    if filtered_users != original_users:

        missing = original_users - filtered_users
        extra = filtered_users - original_users

        raise ValueError(
            f"{dataset_name}: user mismatch after filtering. "
            f"Missing={len(missing):,}, "
            f"Extra={len(extra):,}"
        )

    if filtered_users != source_users:

        raise ValueError(
            f"{dataset_name}: source and filtered user sets differ."
        )

    # ------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------

    users_at_or_below_threshold = 0
    users_above_threshold = 0

    synthetic_kept = 0
    synthetic_removed = 0

    errors = []

    # ------------------------------------------------------------
    # User-by-user validation
    # ------------------------------------------------------------

    for user_id in original:

        original_history = original[user_id]
        source_history = source_augmented[user_id]
        filtered_history = filtered[user_id]

        original_length = len(original_history)

        source_synthetic_count = count_synthetic(
            source_history
        )

        filtered_synthetic_count = count_synthetic(
            filtered_history
        )

        # --------------------------------------------------------
        # Synthetic interaction count
        # --------------------------------------------------------

        if original_length <= threshold:

            users_at_or_below_threshold += 1

            if filtered_synthetic_count != EXPECTED_SYNTHETIC_COUNT:

                errors.append(
                    f"{user_id}: expected "
                    f"{EXPECTED_SYNTHETIC_COUNT} synthetic "
                    f"interactions but found "
                    f"{filtered_synthetic_count}."
                )

        else:

            users_above_threshold += 1

            if filtered_synthetic_count != 0:

                errors.append(
                    f"{user_id}: expected 0 synthetic "
                    f"interactions but found "
                    f"{filtered_synthetic_count}."
                )

        synthetic_kept += filtered_synthetic_count

        synthetic_removed += (
            source_synthetic_count
            - filtered_synthetic_count
        )

        # --------------------------------------------------------
        # Verify every original interaction remains
        # --------------------------------------------------------

        original_keys = Counter(
            interaction_key(interaction)
            for interaction in original_history
        )

        filtered_real_history = get_real_interactions(
            filtered_history
        )

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

        # --------------------------------------------------------
        # Verify validation/test remain unchanged
        # --------------------------------------------------------

        original_real_history = get_real_interactions(
            original_history
        )

        filtered_real_history = get_real_interactions(
            filtered_history
        )

        if len(original_real_history) >= 2:

            original_validation = original_real_history[-2]
            original_test = original_real_history[-1]

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
                        f"{user_id}: validation interaction "
                        "changed."
                    )

                if filtered_test != original_test:

                    errors.append(
                        f"{user_id}: test interaction changed."
                    )

    # ------------------------------------------------------------
    # Report errors
    # ------------------------------------------------------------

    if errors:

        print()
        print("VALIDATION ERRORS:")

        for error in errors[:20]:
            print(f"  - {error}")

        if len(errors) > 20:
            print(
                f"  ... and {len(errors) - 20:,} "
                "more errors."
            )

        raise ValueError(
            f"{dataset_name}: validation FAILED with "
            f"{len(errors):,} error(s)."
        )

    # ------------------------------------------------------------
    # Successful validation summary
    # ------------------------------------------------------------

    print(
        f"Users: "
        f"{len(filtered):,}"
    )

    print(
        f"Users <= threshold: "
        f"{users_at_or_below_threshold:,}"
    )

    print(
        f"Users > threshold: "
        f"{users_above_threshold:,}"
    )

    print(
        f"Synthetic interactions kept: "
        f"{synthetic_kept:,}"
    )

    print(
        f"Synthetic interactions removed: "
        f"{synthetic_removed:,}"
    )

    print()
    print("Filtered validation: PASSED")


def save_json(data, path: Path):
    """
    Save the filtered dataset as a new JSON file.
    """

    print()
    print(f"Saving: {path}")

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved successfully.")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a selective Video Games LLM augmentation "
            "dataset from one existing augmented dataset."
        )
    )

    parser.add_argument(
        "--original",
        required=True,
        help=(
            "Path to the original Video Games "
            "user histories JSON."
        ),
    )

    parser.add_argument(
        "--augmented-input",
        required=True,
        help=(
            "Path to ONE full augmented user histories JSON "
            "(Qwen or LLaMA)."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Path for the new selective augmentation "
            "JSON output."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=7,
        help=(
            "Maximum original history length for retaining "
            "the 3 synthetic interactions. Default: 7."
        ),
    )

    args = parser.parse_args()

    if args.threshold < 0:
        raise ValueError(
            "Threshold must be >= 0."
        )

    original_path = Path(args.original)
    augmented_input_path = Path(
        args.augmented_input
    )
    output_path = Path(args.output)

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    print("=" * 70)
    print("VIDEO GAMES — SELECTIVE LLM AUGMENTATION")
    print("=" * 70)

    print(
        f"Threshold: original history <= "
        f"{args.threshold}"
    )

    print(
        f"Input augmented dataset: "
        f"{augmented_input_path.name}"
    )

    print(
        f"Output: "
        f"{output_path.name}"
    )

    # ------------------------------------------------------------
    # Load original dataset
    # ------------------------------------------------------------

    original = load_json(
        original_path
    )

    print(
        f"\nOriginal users loaded: "
        f"{len(original):,}"
    )

    # ------------------------------------------------------------
    # Load ONE augmented dataset
    # ------------------------------------------------------------

    augmented = load_json(
        augmented_input_path
    )

    print(
        f"Augmented users loaded: "
        f"{len(augmented):,}"
    )

    # ------------------------------------------------------------
    # Validate source
    # ------------------------------------------------------------

    validate_source_dataset(
        original=original,
        augmented=augmented,
        dataset_name=augmented_input_path.name,
    )

    # ------------------------------------------------------------
    # Create selective dataset
    # ------------------------------------------------------------

    filtered = filter_dataset(
        original=original,
        augmented=augmented,
        threshold=args.threshold,
    )

    # ------------------------------------------------------------
    # Validate selective dataset
    # ------------------------------------------------------------

    validate_filtered_dataset(
        original=original,
        source_augmented=augmented,
        filtered=filtered,
        threshold=args.threshold,
        dataset_name=output_path.name,
    )

    # ------------------------------------------------------------
    # Save ONLY after validation passes
    # ------------------------------------------------------------

    save_json(
        data=filtered,
        path=output_path,
    )

    # ------------------------------------------------------------
    # Final message
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("SELECTIVE AUGMENTATION COMPLETE")
    print("=" * 70)

    print(
        f"\nCreated:\n"
        f"  {output_path}"
    )

    print(
        "\nRule applied:"
        f"\n  Original history <= {args.threshold}"
        "  -> keep 3 synthetic interactions"
        f"\n  Original history > {args.threshold}"
        "   -> remove all synthetic interactions"
    )

    print(
        "\nAll validation checks passed."
    )


if __name__ == "__main__":
    main()
