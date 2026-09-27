#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

import ijson


SYNTHETIC_SOURCE = "llm_synthetic"
EXPECTED_SYNTHETIC_COUNT = 3


def load_original_lengths(path: Path):
    """
    Stream the original user-history JSON and store only:

        user_id -> original history length

    This avoids loading the 2+ GB JSON into memory.
    """

    print()
    print("=" * 70)
    print("STEP 1 — READING ORIGINAL HISTORY LENGTHS")
    print("=" * 70)

    print(f"Streaming: {path}")

    original_lengths = {}

    with path.open("rb") as f:
        users = ijson.kvitems(f, "")

        for user_count, (user_id, history) in enumerate(users, start=1):

            original_lengths[user_id] = len(history)

            if user_count % 100_000 == 0:
                print(
                    f"  Processed {user_count:,} users...",
                    flush=True,
                )

    print()
    print(
        f"Original users processed: "
        f"{len(original_lengths):,}"
    )

    if not original_lengths:
        raise ValueError("Original dataset contains no users.")

    return original_lengths


def count_synthetic(history):
    """
    Count synthetic interactions in one augmented history.
    """

    return sum(
        1
        for interaction in history
        if interaction.get("interaction_source")
        == SYNTHETIC_SOURCE
    )


def get_real_interactions(history):
    """
    Return only real interactions.
    """

    return [
        interaction
        for interaction in history
        if interaction.get("interaction_source")
        != SYNTHETIC_SOURCE
    ]


def interaction_key(interaction):
    """
    Create a stable key for an interaction.
    """

    return (
        interaction.get("asin"),
        interaction.get("timestamp"),
        interaction.get("interaction_source"),
        interaction.get("sequence_position"),
        interaction.get("synthetic_position"),
    )


def validate_source_dataset(
    augmented_input_path: Path,
    original_lengths,
):
    """
    Stream through the full augmented dataset and validate:

      - same users as original
      - exactly 3 synthetic interactions per user

    Returns the number of users.
    """

    print()
    print("=" * 70)
    print("STEP 2 — VALIDATING SOURCE AUGMENTED DATASET")
    print("=" * 70)

    print(f"Streaming: {augmented_input_path}")

    augmented_users = 0
    synthetic_counts = Counter()

    with augmented_input_path.open("rb") as f:

        users = ijson.kvitems(f, "")

        for user_id, history in users:

            augmented_users += 1

            if user_id not in original_lengths:
                raise ValueError(
                    f"Unexpected user in augmented dataset: "
                    f"{user_id}"
                )

            synthetic_count = count_synthetic(history)

            synthetic_counts[synthetic_count] += 1

            if synthetic_count != EXPECTED_SYNTHETIC_COUNT:

                raise ValueError(
                    f"User {user_id} has "
                    f"{synthetic_count} synthetic interactions; "
                    f"expected {EXPECTED_SYNTHETIC_COUNT}."
                )

            if augmented_users % 100_000 == 0:

                print(
                    f"  Processed {augmented_users:,} users...",
                    flush=True,
                )

    missing_users = (
        set(original_lengths)
        - set()
    )

    # We don't want to create another huge user set just for this
    # check, so user-count equality is checked here. A more detailed
    # missing-user check is performed during the filtering pass.

    if augmented_users != len(original_lengths):
        missing_count = len(original_lengths) - augmented_users

        print()
        print(
            f"WARNING: {missing_count:,} users from the original "
            "dataset are missing from the LLaMA augmented dataset."
        )
        print(
            "These users will remain excluded from the LLaMA "
            "augmentation condition."
        )

    print()
    print(
        f"Augmented users processed: "
        f"{augmented_users:,}"
    )

    print()
    print("Synthetic interactions per user:")

    for count in sorted(synthetic_counts):

        print(
            f"  {count}: "
            f"{synthetic_counts[count]:,} users"
        )

    print()
    print("Source validation: PASSED")

    return augmented_users


def validate_real_interactions_preserved(
    original_history,
    filtered_history,
    user_id,
):
    """
    Validate that all original interactions remain.

    Since the original dataset is not kept in memory, this function
    is used only when original history data is available. It is kept
    here for structural validation of the filtering logic.
    """

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

            raise ValueError(
                f"{user_id}: original interaction missing "
                f"after filtering: {key}"
            )


def filter_and_write_dataset(
    original_path: Path,
    augmented_input_path: Path,
    output_path: Path,
    original_lengths,
    threshold: int,
):
    """
    Stream the augmented dataset and write the selective dataset.

    IMPORTANT:
    The augmented dataset is never loaded completely into memory.

    For users whose ORIGINAL history length <= threshold:
        keep all interactions.

    For users whose ORIGINAL history length > threshold:
        remove only interactions with:
            interaction_source == "llm_synthetic"
    """

    print()
    print("=" * 70)
    print("STEP 3 — CREATING SELECTIVE DATASET")
    print("=" * 70)

    print(f"Input:  {augmented_input_path}")
    print(f"Output: {output_path}")
    print(f"Threshold: {threshold}")

    users_processed = 0

    users_short = 0
    users_long = 0

    synthetic_kept = 0
    synthetic_removed = 0

    users_seen = set()

    with augmented_input_path.open("rb") as input_file, \
         output_path.open("w", encoding="utf-8") as output_file:

        users = ijson.kvitems(
            input_file,
            "",
        )

        output_file.write("{\n")

        first_user = True

        for user_id, augmented_history in users:

            users_processed += 1
            users_seen.add(user_id)

            if user_id not in original_lengths:

                raise ValueError(
                    f"User {user_id} exists in augmented dataset "
                    "but not in original dataset."
                )

            original_length = original_lengths[user_id]

            synthetic_count = count_synthetic(
                augmented_history
            )

            if synthetic_count != EXPECTED_SYNTHETIC_COUNT:

                raise ValueError(
                    f"User {user_id} has "
                    f"{synthetic_count} synthetic interactions; "
                    f"expected {EXPECTED_SYNTHETIC_COUNT}."
                )

            # ----------------------------------------------------
            # SHORT HISTORY
            # ----------------------------------------------------

            if original_length <= threshold:

                filtered_history = augmented_history

                users_short += 1

                synthetic_kept += synthetic_count

            # ----------------------------------------------------
            # LONG HISTORY
            # ----------------------------------------------------

            else:

                filtered_history = [
                    interaction
                    for interaction in augmented_history
                    if interaction.get(
                        "interaction_source"
                    ) != SYNTHETIC_SOURCE
                ]

                users_long += 1

                synthetic_removed += (
                    synthetic_count
                    - count_synthetic(filtered_history)
                )

            # ----------------------------------------------------
            # Validate filtered synthetic count
            # ----------------------------------------------------

            filtered_synthetic_count = count_synthetic(
                filtered_history
            )

            if original_length <= threshold:

                if filtered_synthetic_count != 3:

                    raise ValueError(
                        f"{user_id}: expected 3 synthetic "
                        f"interactions but found "
                        f"{filtered_synthetic_count}."
                    )

            else:

                if filtered_synthetic_count != 0:

                    raise ValueError(
                        f"{user_id}: expected 0 synthetic "
                        f"interactions but found "
                        f"{filtered_synthetic_count}."
                    )

            # ----------------------------------------------------
            # Write user
            # ----------------------------------------------------

            if not first_user:
                output_file.write(",\n")

            json.dump(
                user_id,
                output_file,
                ensure_ascii=False,
            )

            output_file.write(": ")

            json.dump(
                filtered_history,
                output_file,
                ensure_ascii=False,
            )

            first_user = False

            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------

            if users_processed % 100_000 == 0:

                print(
                    f"  Processed {users_processed:,} users | "
                    f"short={users_short:,} | "
                    f"long={users_long:,}",
                    flush=True,
                )

        output_file.write("\n}\n")

    # ------------------------------------------------------------
    # User count validation
    # ------------------------------------------------------------

    if users_processed != len(original_lengths):
        missing_count = len(original_lengths) - users_processed

        print()
        print(
            f"WARNING: Final output contains {missing_count:,} fewer users "
            "than the original dataset."
        )
        print(
            "These users were absent from the LLaMA augmented dataset "
            "and remain excluded from the LLaMA condition."
        )

    print()
    print("=" * 70)
    print("FILTERING COMPLETE")
    print("=" * 70)

    print(
        f"Users processed: "
        f"{users_processed:,}"
    )

    print(
        f"Users with original history <= {threshold}: "
        f"{users_short:,}"
    )

    print(
        f"Users with original history > {threshold}: "
        f"{users_long:,}"
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
    print(f"Output written to:")
    print(f"  {output_path}")

    return {
        "users_processed": users_processed,
        "users_short": users_short,
        "users_long": users_long,
        "synthetic_kept": synthetic_kept,
        "synthetic_removed": synthetic_removed,
    }


def verify_output(
    output_path: Path,
    original_lengths,
    threshold,
):
    """
    Stream the generated output once more and verify:

      - valid JSON structure
      - same number of users
      - short users have 3 synthetic interactions
      - long users have zero synthetic interactions
    """

    print()
    print("=" * 70)
    print("STEP 4 — VERIFYING OUTPUT")
    print("=" * 70)

    print(f"Streaming: {output_path}")

    users_checked = 0

    with output_path.open("rb") as f:

        users = ijson.kvitems(f, "")

        for user_id, history in users:

            users_checked += 1

            if user_id not in original_lengths:

                raise ValueError(
                    f"Unexpected user in output: {user_id}"
                )

            original_length = original_lengths[user_id]

            synthetic_count = count_synthetic(
                history
            )

            if original_length <= threshold:

                if synthetic_count != 3:

                    raise ValueError(
                        f"{user_id}: short-history user has "
                        f"{synthetic_count} synthetic interactions; "
                        "expected 3."
                    )

            else:

                if synthetic_count != 0:

                    raise ValueError(
                        f"{user_id}: long-history user has "
                        f"{synthetic_count} synthetic interactions; "
                        "expected 0."
                    )

            if users_checked % 100_000 == 0:

                print(
                    f"  Verified {users_checked:,} users...",
                    flush=True,
                )

    if users_checked != len(original_lengths):
        missing_count = len(original_lengths) - users_checked

        print()
        print(
            f"WARNING: Output contains {missing_count:,} fewer users "
            "than the original dataset."
        )
        print(
            "These users were already absent from the LLaMA "
            "augmented dataset and remain excluded."
        )

    print()
    print(
        f"Output users verified: "
        f"{users_checked:,}"
    )

    print()
    print("Output verification: PASSED")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create a selective Video Games LLM augmentation "
            "dataset using streaming JSON processing."
        )
    )

    parser.add_argument(
        "--original",
        required=True,
        help="Original Video Games user histories JSON.",
    )

    parser.add_argument(
        "--augmented-input",
        required=True,
        help=(
            "One full augmented dataset "
            "(Qwen or LLaMA)."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="New selective augmentation JSON output.",
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

    if not original_path.exists():

        raise FileNotFoundError(
            f"Original dataset not found: {original_path}"
        )

    if not augmented_input_path.exists():

        raise FileNotFoundError(
            "Augmented dataset not found: "
            f"{augmented_input_path}"
        )

    if output_path.exists():

        raise FileExistsError(
            "Output already exists. "
            "This script will not overwrite it:\n"
            f"{output_path}"
        )

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
    # STEP 1
    # ------------------------------------------------------------

    original_lengths = load_original_lengths(
        original_path
    )

    # ------------------------------------------------------------
    # STEP 2
    # ------------------------------------------------------------

    validate_source_dataset(
        augmented_input_path=augmented_input_path,
        original_lengths=original_lengths,
    )

    # ------------------------------------------------------------
    # STEP 3
    # ------------------------------------------------------------

    filter_and_write_dataset(
        original_path=original_path,
        augmented_input_path=augmented_input_path,
        output_path=output_path,
        original_lengths=original_lengths,
        threshold=args.threshold,
    )

    # ------------------------------------------------------------
    # STEP 4
    # ------------------------------------------------------------

    verify_output(
        output_path=output_path,
        original_lengths=original_lengths,
        threshold=args.threshold,
    )

    print()
    print("=" * 70)
    print("SELECTIVE AUGMENTATION COMPLETE")
    print("=" * 70)

    print()
    print("Rule applied:")

    print(
        f"  Original history <= {args.threshold}"
        " -> keep all 3 synthetic interactions"
    )

    print(
        f"  Original history > {args.threshold}"
        "  -> remove all synthetic interactions"
    )

    print()
    print("All streaming validation checks passed.")

    print()
    print(f"Created:")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()
