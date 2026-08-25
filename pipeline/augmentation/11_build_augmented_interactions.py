"""
11_build_augmented_interactions.py


IMPORTANT:
    This script does NOT invent timestamps for synthetic items.

    It stores the augmented sequence order explicitly.
    Step 12 will convert the sequence into RecBole format using
    deterministic sequence-position order values so that
    validation and test remain the final two interactions.

Output:
    {prefix}_augmented_user_histories.json



    python pipeline/augmentation/11_build_augmented_interactions.py --dataset fashion

    python pipeline/augmentation/11_build_augmented_interactions.py --dataset video_games

    python pipeline/augmentation/11_build_augmented_interactions.py --dataset movies_tv
"""

import json
import os
import argparse


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
            "Build augmented user interaction histories "
            "for an Amazon dataset."
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

    return parser.parse_args()


# ==========================================================
# Helpers
# ==========================================================

def clean_text(value):
    """
    Convert a value into a clean string.
    """

    if value is None:
        return ""

    return str(value).strip()


def load_json(path):
    """
    Load a JSON file with a clear error if it does not exist.
    """

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
# Build Augmented Histories
# ==========================================================

def build_augmented_histories():

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

    user_histories_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_user_histories.json"
    )

    augmentation_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llama_augmentations.json"
    )

    catalogue_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_catalogue.json"
    )

    output_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llama_augmented_user_histories.json"
    )

    # ======================================================
    # Header
    # ======================================================

    print("=" * 70)

    print(
        f"BUILDING "
        f"{dataset_folder.upper()} "
        f"AUGMENTED USER HISTORIES"
    )

    print("=" * 70)

    # ======================================================
    # Load Original Histories
    # ======================================================

    print()

    original_histories = load_json(
        user_histories_file
    )

    print(
        f"Original users: "
        f"{len(original_histories):,}"
    )

    # ======================================================
    # Load Validated LLM Augmentations
    # ======================================================

    print()

    augmentation_records = load_json(
        augmentation_file
    )

    print(
        f"Validated augmentation users: "
        f"{len(augmentation_records):,}"
    )

    # ======================================================
    # Load Catalogue
    # ======================================================

    print()

    catalogue = load_json(
        catalogue_file
    )

    print(
        f"Catalogue products: "
        f"{len(catalogue):,}"
    )

    # ======================================================
    # Build Lookups
    # ======================================================

    augmentation_lookup = {
        clean_text(
            record.get("user_id")
        ): record

        for record in augmentation_records

        if clean_text(
            record.get("user_id")
        )
    }

    catalogue_lookup = {
        clean_text(
            product.get("asin")
        ): product

        for product in catalogue

        if clean_text(
            product.get("asin")
        )
    }

    # ======================================================
    # Counters
    # ======================================================

    augmented_histories = {}

    users_written = 0
    users_missing_augmentation = 0
    users_invalid_history = 0

    original_interactions = 0
    synthetic_interactions = 0
    final_interactions = 0

    duplicate_synthetic_asins = 0
    overlap_with_train = 0

    validation_changed = 0
    test_changed = 0

    # ======================================================
    # Process Users
    # ======================================================

    for user_id, history in (
        original_histories.items()
    ):

        user_id = clean_text(
            user_id
        )

        # --------------------------------------------------
        # Defensive chronological sort
        # --------------------------------------------------

        history = sorted(
            history,
            key=lambda item: (
                item.get("timestamp")
                if item.get("timestamp")
                is not None
                else 0
            )
        )

        full_length = len(
            history
        )

        original_interactions += (
            full_length
        )

        # Need:
        #   >= 1 train
        #   1 validation
        #   1 test

        if full_length < 3:

            users_invalid_history += 1
            continue

        # --------------------------------------------------
        # Original fixed split
        # --------------------------------------------------

        train_history = history[:-2]

        validation_item = history[-2]

        test_item = history[-1]

        # --------------------------------------------------
        # Get validated augmentations
        # --------------------------------------------------

        augmentation_record = (
            augmentation_lookup.get(
                user_id
            )
        )

        if augmentation_record is None:

            users_missing_augmentation += 1
            continue

        augmented_asins = (
            augmentation_record.get(
                "augmented_asins",
                []
            )
        )

        if (
            len(augmented_asins)
            != EXPECTED_AUGMENTATIONS
        ):

            raise RuntimeError(
                f"\nUser {user_id} does not have "
                f"exactly {EXPECTED_AUGMENTATIONS} "
                "validated augmentation ASINs."
            )

        augmented_asins = [
            clean_text(asin)
            for asin in augmented_asins
        ]

        # --------------------------------------------------
        # Duplicate augmentation check
        # --------------------------------------------------

        if (
            len(augmented_asins)
            != len(
                set(augmented_asins)
            )
        ):

            duplicate_synthetic_asins += 1

            raise RuntimeError(
                f"\nDuplicate synthetic ASINs found "
                f"for user {user_id}"
            )

        # --------------------------------------------------
        # Training-history overlap check
        # --------------------------------------------------

        train_asins = {
            clean_text(
                item.get("asin")
            )

            for item in train_history

            if clean_text(
                item.get("asin")
            )
        }

        overlapping = (
            set(augmented_asins)
            & train_asins
        )

        if overlapping:

            overlap_with_train += 1

            raise RuntimeError(
                "\nSynthetic item overlaps with "
                "original training history.\n"
                f"User: {user_id}\n"
                f"ASINs: {sorted(overlapping)}"
            )

        # --------------------------------------------------
        # Build synthetic interaction records
        # --------------------------------------------------

        synthetic_items = []

        for synthetic_position, asin in enumerate(
            augmented_asins,
            start=1
        ):

            product = (
                catalogue_lookup.get(
                    asin
                )
            )

            if product is None:

                raise RuntimeError(
                    f"\nSynthetic ASIN {asin} "
                    f"is missing from the "
                    f"{dataset_folder} catalogue."
                )

            synthetic_items.append(
                {
                    "asin": (
                        asin
                    ),

                    # No real timestamp exists.
                    # Step 12 assigns deterministic
                    # sequence-based order values.
                    "timestamp": None,

                    "title": clean_text(
                        product.get(
                            "title"
                        )
                    ),

                    "store": clean_text(
                        product.get(
                            "store"
                        )
                    ),

                    "main_category": clean_text(
                        product.get(
                            "main_category"
                        )
                    ),

                    "categories": product.get(
                        "categories",
                        []
                    ),

                    "department": clean_text(
                        product.get(
                            "department"
                        )
                    ),

                    "manufacturer": clean_text(
                        product.get(
                            "manufacturer"
                        )
                    ),

                    "description": clean_text(
                        product.get(
                            "description"
                        )
                    ),

                    "features": product.get(
                        "features",
                        []
                    ),

                    "interaction_source": (
                        "llm_synthetic"
                    ),

                    "synthetic_position": (
                        synthetic_position
                    ),
                }
            )

        # --------------------------------------------------
        # Mark Original Interactions
        # --------------------------------------------------

        marked_train = []

        for item in train_history:

            marked_item = dict(
                item
            )

            marked_item[
                "interaction_source"
            ] = "real_train"

            marked_train.append(
                marked_item
            )

        marked_validation = dict(
            validation_item
        )

        marked_validation[
            "interaction_source"
        ] = "real_validation"

        marked_test = dict(
            test_item
        )

        marked_test[
            "interaction_source"
        ] = "real_test"

        # --------------------------------------------------
        # Construct Augmented Sequence
        # --------------------------------------------------

        augmented_sequence = (
            marked_train
            + synthetic_items
            + [
                marked_validation,
                marked_test,
            ]
        )

        # --------------------------------------------------
        # Add explicit sequence positions
        # --------------------------------------------------

        for position, item in enumerate(
            augmented_sequence,
            start=1
        ):

            item[
                "sequence_position"
            ] = position

        # ==================================================
        # Critical Validation
        # ==================================================

        # --------------------------------------------------
        # Validation item must remain unchanged
        # --------------------------------------------------

        if (
            clean_text(
                augmented_sequence[-2].get(
                    "asin"
                )
            )
            != clean_text(
                validation_item.get(
                    "asin"
                )
            )
        ):

            validation_changed += 1

            raise RuntimeError(
                f"Validation item changed "
                f"for user {user_id}"
            )

        # --------------------------------------------------
        # Test item must remain unchanged
        # --------------------------------------------------

        if (
            clean_text(
                augmented_sequence[-1].get(
                    "asin"
                )
            )
            != clean_text(
                test_item.get(
                    "asin"
                )
            )
        ):

            test_changed += 1

            raise RuntimeError(
                f"Test item changed "
                f"for user {user_id}"
            )

        # --------------------------------------------------
        # Exactly EXPECTED_AUGMENTATIONS synthetic items
        # --------------------------------------------------

        synthetic_count = sum(
            1
            for item in augmented_sequence
            if item.get(
                "interaction_source"
            ) == "llm_synthetic"
        )

        if (
            synthetic_count
            != EXPECTED_AUGMENTATIONS
        ):

            raise RuntimeError(
                f"User {user_id} has "
                f"{synthetic_count} synthetic items "
                f"instead of "
                f"{EXPECTED_AUGMENTATIONS}."
            )

        # --------------------------------------------------
        # Final length check
        # --------------------------------------------------

        expected_final_length = (
            full_length
            + EXPECTED_AUGMENTATIONS
        )

        if (
            len(augmented_sequence)
            != expected_final_length
        ):

            raise RuntimeError(
                f"Augmented length mismatch "
                f"for user {user_id}"
            )

        # --------------------------------------------------
        # Save User
        # --------------------------------------------------

        augmented_histories[
            user_id
        ] = augmented_sequence

        users_written += 1

        synthetic_interactions += (
            EXPECTED_AUGMENTATIONS
        )

        final_interactions += len(
            augmented_sequence
        )

    # ======================================================
    # Global Validation
    # ======================================================

    print()
    print("=" * 70)

    print(
        "VALIDATING AUGMENTED HISTORIES"
    )

    print("=" * 70)

    if not augmented_histories:

        raise RuntimeError(
            "No augmented histories were created."
        )

    for user_id, sequence in (
        augmented_histories.items()
    ):

        # Last two must remain validation/test

        if (
            sequence[-2].get(
                "interaction_source"
            )
            != "real_validation"
        ):

            raise RuntimeError(
                f"Validation position corrupted "
                f"for user {user_id}"
            )

        if (
            sequence[-1].get(
                "interaction_source"
            )
            != "real_test"
        ):

            raise RuntimeError(
                f"Test position corrupted "
                f"for user {user_id}"
            )

        # Synthetic items must appear BEFORE validation/test

        synthetic_positions = [
            item[
                "sequence_position"
            ]

            for item in sequence

            if item.get(
                "interaction_source"
            ) == "llm_synthetic"
        ]

        validation_position = (
            sequence[-2][
                "sequence_position"
            ]
        )

        if any(
            position >= validation_position
            for position in synthetic_positions
        ):

            raise RuntimeError(
                f"Synthetic interaction appears "
                f"after validation boundary "
                f"for user {user_id}"
            )

        # --------------------------------------------------
        # Sequence positions must be contiguous
        # --------------------------------------------------

        actual_positions = [
            item.get(
                "sequence_position"
            )
            for item in sequence
        ]

        expected_positions = list(
            range(
                1,
                len(sequence) + 1
            )
        )

        if (
            actual_positions
            != expected_positions
        ):

            raise RuntimeError(
                f"Non-contiguous sequence positions "
                f"for user {user_id}"
            )

    print(
        "Validation passed."
    )

    print(
        "All synthetic items occur before "
        "validation/test."
    )

    print(
        "All validation interactions remain "
        "second-last."
    )

    print(
        "All test interactions remain last."
    )

    print(
        "All sequence positions are contiguous."
    )

    # ======================================================
    # Save Output
    # ======================================================

    print()
    print(
        "Saving augmented user histories:"
    )

    print(
        output_file
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            augmented_histories,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ======================================================
    # Summary
    # ======================================================

    print()
    print("=" * 70)

    print(
        f"{dataset_folder.upper()} "
        f"AUGMENTED HISTORIES CREATED"
    )

    print("=" * 70)

    print(
        f"Dataset:                     "
        f"{dataset_folder}"
    )

    print(
        f"Original users:              "
        f"{len(original_histories):,}"
    )

    print(
        f"Users written:               "
        f"{users_written:,}"
    )

    print(
        f"Users missing augmentation:  "
        f"{users_missing_augmentation:,}"
    )

    print(
        f"Users with invalid history:  "
        f"{users_invalid_history:,}"
    )

    print()

    print(
        f"Original interactions:       "
        f"{original_interactions:,}"
    )

    print(
        f"Synthetic interactions:      "
        f"{synthetic_interactions:,}"
    )

    print(
        f"Final interactions:          "
        f"{final_interactions:,}"
    )

    print()

    print(
        f"Duplicate synthetic sets:    "
        f"{duplicate_synthetic_asins:,}"
    )

    print(
        f"Train-history overlap:       "
        f"{overlap_with_train:,}"
    )

    print(
        f"Validation items changed:    "
        f"{validation_changed:,}"
    )

    print(
        f"Test items changed:          "
        f"{test_changed:,}"
    )

    # ======================================================
    # Example User
    # ======================================================

    example_user = next(
        iter(
            augmented_histories
        )
    )

    example_sequence = (
        augmented_histories[
            example_user
        ]
    )

    print()
    print("=" * 70)

    print(
        "EXAMPLE AUGMENTED USER"
    )

    print("=" * 70)

    print(
        f"User ID:\n"
        f"{example_user}"
    )

    print()

    print(
        f"Final sequence length: "
        f"{len(example_sequence)}"
    )

    print()

    print(
        "Sequence:"
    )

    for item in example_sequence:

        print(
            f"{item['sequence_position']:>3}. "
            f"{item.get('asin')} "
            f"[{item.get('interaction_source')}]"
        )

    print()

    print(
        f"Output file:\n"
        f"{output_file}"
    )

    print()
    print("=" * 70)

    print(
        "COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        "Next step:"
    )

    print(
        "12_export_augmented_recbole.py "
        f"--dataset {dataset_key}"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    build_augmented_histories()