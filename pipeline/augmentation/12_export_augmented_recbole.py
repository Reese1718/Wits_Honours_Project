"""
12_export_augmented_recbole.py


    python pipeline/augmentation/12_export_augmented_recbole.py --dataset fashion

    python pipeline/augmentation/12_export_augmented_recbole.py --dataset video_games

    python pipeline/augmentation/12_export_augmented_recbole.py --dataset movies_tv
"""

import json
import os
import argparse
import ijson


# ==========================================================
# Dataset Configuration
# ==========================================================

DATASETS = {
    "fashion": {
        "folder": "Fashion",
        "prefix": "Fashion",
    },

    "video_games": {
        "folder": "Video Games",
        "prefix": "Video_Games",
    },

    "movies_tv": {
        "folder": "Movies and TV Shows",
        "prefix": "Movies_TV",
    },
}


# ==========================================================
# Configuration
# ==========================================================

EXPECTED_AUGMENTATIONS = 3


# ==========================================================
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Export augmented interaction histories "
            "to RecBole .inter format."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
        help=(
            "Dataset to process: "
            "fashion, video_games, or movies_tv"
        ),
    )

    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Optional custom input JSON file. "
            "If omitted, the default LLaMA augmented file is used."
        ),
    )

    parser.add_argument(
        "--output-file",
        default=None,
        help=(
            "Optional custom output .inter file. "
            "If omitted, the default LLaMA augmented file is used."
        ),
    )

    parser.add_argument(
        "--selective",
        action="store_true",
        help=(
            "Indicate that the input is a selective augmentation "
            "dataset, where users may have 0 or 3 synthetic interactions."
        ),
    )

    return parser.parse_args()


# ==========================================================
# Helpers
# ==========================================================

def clean_text(value):

    if value is None:
        return ""

    return str(value).strip()


def load_json(path):

    print("Loading:")
    print(path)

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ==========================================================
# Export
# ==========================================================

def export_augmented_recbole():

    args = parse_args()

    dataset_key = args.dataset
    config = DATASETS[dataset_key]

    dataset_folder = config["folder"]
    dataset_prefix = config["prefix"]

    # ======================================================
    # Project Paths
    # ======================================================

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_dir = os.path.dirname(
        os.path.dirname(script_dir)
    )

    export_dir = os.path.join(
        project_dir,
        "datasets",
        dataset_folder,
        "exports"
    )

    input_file = (
        args.input_file
        if args.input_file is not None
        else os.path.join(
            export_dir,
            f"{dataset_prefix}_llama_augmented_user_histories.json"
        )
    )

    output_file = (
        args.output_file
        if args.output_file is not None
        else os.path.join(
            export_dir,
            f"{dataset_prefix}_llama_augmented.inter"
        )
    )

    # ======================================================
    # Header
    # ======================================================

    print("=" * 70)

    print(
        f"EXPORTING AUGMENTED "
        f"{dataset_folder.upper()} "
        f"DATA TO RECBOLE"
    )

    print("=" * 70)

    # ======================================================
    # Load Histories
    # ======================================================

    print()

    print(
        "Streaming user histories..."
    )

    # ======================================================
    # Validation Counters
    # ======================================================

    total_interactions = 0

    real_train_count = 0
    synthetic_count = 0
    validation_count = 0
    test_count = 0

    invalid_users = 0
    invalid_positions = 0

    rows = []

    # ======================================================
    # Process Users
    # ======================================================

    user_count = 0

    with open(
        input_file,
        "rb"
    ) as file:

        user_histories = ijson.kvitems(
            file,
            ""
        )

        for user_id, sequence in user_histories:

            user_count += 1

            user_id = clean_text(
                user_id
            )

            if not user_id:

                invalid_users += 1
                continue

            if not sequence:

                invalid_users += 1
                continue

            # --------------------------------------------------
            # Sort strictly by explicit sequence_position
            # --------------------------------------------------

            sequence = sorted(
                sequence,
                key=lambda item: (
                    item.get(
                        "sequence_position",
                        0
                    )
                )
            )

            # --------------------------------------------------
            # Validate sequence positions
            # --------------------------------------------------

            expected_positions = list(
                range(
                    1,
                    len(sequence) + 1
                )
            )

            actual_positions = [
                item.get(
                    "sequence_position"
                )
                for item in sequence
            ]

            if (
                actual_positions
                != expected_positions
            ):

                invalid_positions += 1

                raise RuntimeError(
                    "\nInvalid sequence positions found.\n"
                    f"User: {user_id}\n"
                    f"Expected: {expected_positions}\n"
                    f"Actual:   {actual_positions}"
                )

            # --------------------------------------------------
            # Last two positions must be validation and test
            # --------------------------------------------------

            if len(sequence) < 3:

                raise RuntimeError(
                    f"User {user_id} has fewer than "
                    "3 interactions."
                )

            if (
                sequence[-2].get(
                    "interaction_source"
                )
                != "real_validation"
            ):

                raise RuntimeError(
                    "\nValidation boundary corrupted.\n"
                    f"User: {user_id}"
                )

            if (
                sequence[-1].get(
                    "interaction_source"
                )
                != "real_test"
            ):

                raise RuntimeError(
                    "\nTest boundary corrupted.\n"
                    f"User: {user_id}"
                )

            # --------------------------------------------------
            # Export every interaction
            # --------------------------------------------------

            for item in sequence:

                item_id = clean_text(
                    item.get("asin")
                )

                sequence_position = (
                    item.get(
                        "sequence_position"
                    )
                )

                source = clean_text(
                    item.get(
                        "interaction_source"
                    )
                )

                if not item_id:

                    raise RuntimeError(
                        "\nMissing ASIN/item_id.\n"
                        f"User: {user_id}\n"
                        f"Position: {sequence_position}"
                    )

                if sequence_position is None:

                    raise RuntimeError(
                        "\nMissing sequence_position.\n"
                        f"User: {user_id}\n"
                        f"ASIN: {item_id}"
                    )

                # ------------------------------------------------
                # Count provenance
                # ------------------------------------------------

                if source == "real_train":

                    real_train_count += 1

                elif source == "llm_synthetic":

                    synthetic_count += 1

                elif source == "real_validation":

                    validation_count += 1

                elif source == "real_test":

                    test_count += 1

                else:

                    raise RuntimeError(
                        "\nUnknown interaction source.\n"
                        f"User: {user_id}\n"
                        f"Source: {source}"
                    )

                # ------------------------------------------------
                # RecBole row
                #
                # sequence_position becomes the temporal/order field
                # ------------------------------------------------

                rows.append(
                    (
                        user_id,
                        item_id,
                        float(
                            sequence_position
                        )
                    )
                )

                total_interactions += 1

    # ======================================================
    # Global Checks
    # ======================================================

    print()
    print("=" * 70)

    print(
        "VALIDATING RECBOLE EXPORT"
    )

    print("=" * 70)

    if invalid_users > 0:

        raise RuntimeError(
            f"{invalid_users} invalid users detected."
        )

    if invalid_positions > 0:

        raise RuntimeError(
            f"{invalid_positions} users had "
            "invalid sequence positions."
        )

    expected_users = user_count

    if (
        validation_count
        != expected_users
    ):

        raise RuntimeError(
            "\nValidation count mismatch.\n"
            f"Expected: {expected_users:,}\n"
            f"Actual:   {validation_count:,}"
        )

    if (
        test_count
        != expected_users
    ):

        raise RuntimeError(
            "\nTest count mismatch.\n"
            f"Expected: {expected_users:,}\n"
            f"Actual:   {test_count:,}"
        )

    expected_synthetic = (
        expected_users
        * EXPECTED_AUGMENTATIONS
    )

    if (
        synthetic_count
        != expected_synthetic
    ):

        raise RuntimeError(
            "\nSynthetic interaction count mismatch.\n"
            f"Expected: {expected_synthetic:,}\n"
            f"Actual:   {synthetic_count:,}"
        )

    expected_total = (
        real_train_count
        + synthetic_count
        + validation_count
        + test_count
    )

    if (
        total_interactions
        != expected_total
    ):

        raise RuntimeError(
            "Interaction-count reconciliation failed."
        )

    print(
        "Export validation passed."
    )

    print(
        "Every user has exactly one "
        "validation interaction."
    )

    print(
        "Every user has exactly one "
        "test interaction."
    )

    print(
        "Every user has exactly "
        f"{EXPECTED_AUGMENTATIONS} "
        "LLM synthetic interactions."
    )

    # ======================================================
    # Write RecBole .inter File
    # ======================================================

    print()
    print(
        "Writing RecBole interaction file:"
    )

    print(
        output_file
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "user_id:token\t"
            "item_id:token\t"
            "timestamp:float\n"
        )

        for (
            user_id,
            item_id,
            timestamp
        ) in rows:

            file.write(
                f"{user_id}\t"
                f"{item_id}\t"
                f"{timestamp}\n"
            )

    # ======================================================
    # Summary
    # ======================================================

    print()
    print("=" * 70)

    print(
        "AUGMENTED RECBOLE EXPORT COMPLETE"
    )

    print("=" * 70)

    print(
        f"Dataset:                "
        f"{dataset_folder}"
    )

    print(
        f"Users:                  "
        f"{expected_users:,}"
    )

    print(
        f"Real train interactions:"
        f" {real_train_count:,}"
    )

    print(
        f"Synthetic interactions: "
        f"{synthetic_count:,}"
    )

    print(
        f"Validation interactions:"
        f" {validation_count:,}"
    )

    print(
        f"Test interactions:      "
        f"{test_count:,}"
    )

    print(
        f"Total interactions:     "
        f"{total_interactions:,}"
    )

    print()
    print(
        f"Output file:\n"
        f"{output_file}"
    )

    # ======================================================
    # Example Rows
    # ======================================================

    print()
    print("=" * 70)

    print(
        "FIRST 10 EXPORTED ROWS"
    )

    print("=" * 70)

    for row in rows[:10]:

        print(
            row
        )

    print()
    print("=" * 70)

    print(
        "COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        f"The augmented {dataset_folder} dataset "
        "is now ready for RecBole training."
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    export_augmented_recbole()
