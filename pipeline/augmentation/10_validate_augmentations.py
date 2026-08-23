"""
10_validate_augmentations.py

Validates raw LLM augmentation output for one of the supported datasets.

    python pipeline/augmentation/10_validate_augmentations.py --dataset fashion

    python pipeline/augmentation/10_validate_augmentations.py --dataset video_games

    python pipeline/augmentation/10_validate_augmentations.py --dataset movies_tv
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
            "Validate LLM-generated augmentation records "
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
# Main Validation
# ==========================================================

def validate_augmentations():

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

    raw_augmentation_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_augmentations_raw.json"
    )

    train_input_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_train_inputs.json"
    )

    train_candidate_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_train_candidates.json"
    )

    catalogue_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_catalogue.json"
    )

    output_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_augmentations.json"
    )

    invalid_output_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_augmentations_invalid.json"
    )

    # ======================================================
    # Header
    # ======================================================

    print("=" * 70)

    print(
        f"VALIDATING "
        f"{dataset_folder.upper()} "
        f"LLM AUGMENTATIONS"
    )

    print("=" * 70)

    # ======================================================
    # Load Files
    # ======================================================

    print()

    raw_records = load_json(
        raw_augmentation_file
    )

    print(
        f"Raw augmentation records: "
        f"{len(raw_records):,}"
    )

    print()

    train_inputs = load_json(
        train_input_file
    )

    print(
        f"Train-only input users: "
        f"{len(train_inputs):,}"
    )

    print()

    candidate_records = load_json(
        train_candidate_file
    )

    print(
        f"Candidate records: "
        f"{len(candidate_records):,}"
    )

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

    train_lookup = {
        record["user_id"]: record
        for record in train_inputs
        if record.get("user_id")
    }

    candidate_lookup = {
        record["user_id"]: record
        for record in candidate_records
        if record.get("user_id")
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
    # Validation Counters
    # ======================================================

    valid_records = []
    invalid_records = []

    invalid_status = 0
    missing_user = 0
    missing_candidate_record = 0

    wrong_selection_count = 0
    invalid_candidate_id = 0
    asin_mapping_error = 0

    missing_catalogue_asin = 0

    duplicate_candidate_ids = 0
    duplicate_asins = 0

    train_history_overlap = 0

    # ======================================================
    # Validate Every Raw Record
    # ======================================================

    for raw_record in raw_records:

        user_id = clean_text(
            raw_record.get("user_id")
        )

        errors = []

        # --------------------------------------------------
        # User existence
        # --------------------------------------------------

        if not user_id:

            errors.append(
                "Missing user_id"
            )

            missing_user += 1

        elif user_id not in train_lookup:

            errors.append(
                "User does not exist in "
                "train-only LLM inputs"
            )

            missing_user += 1

        # --------------------------------------------------
        # Original generation status
        # --------------------------------------------------

        status = clean_text(
            raw_record.get("status")
        )

        if status != "success":

            errors.append(
                f"Raw generation status "
                f"is '{status}'"
            )

            invalid_status += 1

        # --------------------------------------------------
        # Selected candidates
        # --------------------------------------------------

        selected_candidates = raw_record.get(
            "selected_candidates",
            []
        )

        if not isinstance(
            selected_candidates,
            list
        ):

            selected_candidates = []

        if (
            len(selected_candidates)
            != EXPECTED_AUGMENTATIONS
        ):

            errors.append(
                "Expected exactly "
                f"{EXPECTED_AUGMENTATIONS} "
                "selected candidates, "
                f"found "
                f"{len(selected_candidates)}"
            )

            wrong_selection_count += 1

        # --------------------------------------------------
        # Candidate record
        # --------------------------------------------------

        user_candidate_record = (
            candidate_lookup.get(
                user_id
            )
        )

        if (
            user_candidate_record
            is None
        ):

            errors.append(
                "Candidate record not found"
            )

            missing_candidate_record += 1

            candidate_id_lookup = {}

        else:

            candidate_id_lookup = {
                clean_text(
                    candidate.get(
                        "candidate_id"
                    )
                ): candidate

                for candidate
                in user_candidate_record.get(
                    "candidates",
                    []
                )

                if clean_text(
                    candidate.get(
                        "candidate_id"
                    )
                )
            }

        # --------------------------------------------------
        # Train-history ASINs
        # --------------------------------------------------

        train_record = train_lookup.get(
            user_id,
            {}
        )

        train_history = train_record.get(
            "history",
            []
        )

        train_asins = {
            clean_text(
                product.get("asin")
            )
            for product in train_history
            if clean_text(
                product.get("asin")
            )
        }

        # --------------------------------------------------
        # Duplicate checks
        # --------------------------------------------------

        selected_candidate_ids = [
            clean_text(
                item.get("candidate_id")
            )
            for item in selected_candidates
        ]

        selected_asins = [
            clean_text(
                item.get("asin")
            )
            for item in selected_candidates
        ]

        if (
            len(selected_candidate_ids)
            != len(
                set(selected_candidate_ids)
            )
        ):

            errors.append(
                "Duplicate candidate IDs"
            )

            duplicate_candidate_ids += 1

        if (
            len(selected_asins)
            != len(
                set(selected_asins)
            )
        ):

            errors.append(
                "Duplicate selected ASINs"
            )

            duplicate_asins += 1

        # --------------------------------------------------
        # Validate each selected item
        # --------------------------------------------------

        clean_selected_items = []

        for selected in selected_candidates:

            candidate_id = clean_text(
                selected.get(
                    "candidate_id"
                )
            )

            asin = clean_text(
                selected.get(
                    "asin"
                )
            )

            candidate = (
                candidate_id_lookup.get(
                    candidate_id
                )
            )

            # Candidate ID must exist
            if candidate is None:

                errors.append(
                    f"Invalid candidate ID: "
                    f"{candidate_id}"
                )

                invalid_candidate_id += 1

                continue

            expected_asin = clean_text(
                candidate.get(
                    "asin"
                )
            )

            # Candidate ID -> ASIN mapping
            if asin != expected_asin:

                errors.append(
                    f"Candidate mapping mismatch: "
                    f"{candidate_id} -> "
                    f"{expected_asin}, "
                    f"but raw output contains "
                    f"{asin}"
                )

                asin_mapping_error += 1

                continue

            # Must exist in catalogue
            if asin not in catalogue_lookup:

                errors.append(
                    f"ASIN missing from catalogue: "
                    f"{asin}"
                )

                missing_catalogue_asin += 1

                continue

            # Must NOT already exist in TRAIN history
            if asin in train_asins:

                errors.append(
                    f"Selected ASIN already exists "
                    f"in train history: {asin}"
                )

                train_history_overlap += 1

                continue

            clean_selected_items.append(
                {
                    "candidate_id": (
                        candidate_id
                    ),

                    "asin": (
                        asin
                    ),

                    "title": clean_text(
                        candidate.get(
                            "title"
                        )
                    ),

                    "similarity_score": (
                        candidate.get(
                            "similarity_score"
                        )
                    ),
                }
            )

        # --------------------------------------------------
        # Final Record Decision
        # --------------------------------------------------

        if errors:

            invalid_records.append(
                {
                    "user_id": (
                        user_id
                    ),

                    "errors": (
                        errors
                    ),

                    "raw_model_response": (
                        raw_record.get(
                            "raw_model_response",
                            ""
                        )
                    ),
                }
            )

            continue

        if (
            len(clean_selected_items)
            != EXPECTED_AUGMENTATIONS
        ):

            invalid_records.append(
                {
                    "user_id": (
                        user_id
                    ),

                    "errors": [
                        "Final validated item count "
                        "is not exactly "
                        f"{EXPECTED_AUGMENTATIONS}"
                    ],
                }
            )

            continue

        # --------------------------------------------------
        # Clean Final Augmentation
        # --------------------------------------------------

        valid_records.append(
            {
                "user_id": (
                    user_id
                ),

                "augmented_asins": [
                    item["asin"]
                    for item
                    in clean_selected_items
                ],

                "selected_candidates": (
                    clean_selected_items
                ),
            }
        )

    # ======================================================
    # Global Validation
    # ======================================================

    print()
    print("=" * 70)

    print(
        "GLOBAL VALIDATION"
    )

    print("=" * 70)

    # ------------------------------------------------------
    # Duplicate output users
    # ------------------------------------------------------

    valid_user_ids = [
        record["user_id"]
        for record in valid_records
    ]

    if (
        len(valid_user_ids)
        != len(
            set(valid_user_ids)
        )
    ):

        raise RuntimeError(
            "Duplicate users found in "
            "validated augmentation output."
        )

    print(
        "No duplicate users in valid output."
    )

    # ------------------------------------------------------
    # Every valid record has exactly 3 items
    # ------------------------------------------------------

    for record in valid_records:

        if (
            len(
                record[
                    "augmented_asins"
                ]
            )
            != EXPECTED_AUGMENTATIONS
        ):

            raise RuntimeError(
                "Validated record does not "
                f"contain exactly "
                f"{EXPECTED_AUGMENTATIONS} ASINs."
            )

    print(
        "Every valid record contains "
        f"exactly {EXPECTED_AUGMENTATIONS} "
        "augmentations."
    )

    # ======================================================
    # Save Valid Output
    # ======================================================

    print()
    print(
        "Saving validated augmentations:"
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
            valid_records,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ======================================================
    # Save Invalid Report If Needed
    # ======================================================

    if invalid_records:

        with open(
            invalid_output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                invalid_records,
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
        f"AUGMENTATION VALIDATION COMPLETE"
    )

    print("=" * 70)

    print(
        f"Dataset:                     "
        f"{dataset_folder}"
    )

    print(
        f"Raw records:                 "
        f"{len(raw_records):,}"
    )

    print(
        f"Valid records:               "
        f"{len(valid_records):,}"
    )

    print(
        f"Invalid records:             "
        f"{len(invalid_records):,}"
    )

    print()
    print(
        "Failure breakdown"
    )

    print(
        "-" * 70
    )

    print(
        f"Non-success generation:      "
        f"{invalid_status:,}"
    )

    print(
        f"Missing/invalid user:        "
        f"{missing_user:,}"
    )

    print(
        f"Missing candidate record:    "
        f"{missing_candidate_record:,}"
    )

    print(
        f"Wrong selection count:       "
        f"{wrong_selection_count:,}"
    )

    print(
        f"Invalid candidate IDs:       "
        f"{invalid_candidate_id:,}"
    )

    print(
        f"ASIN mapping errors:         "
        f"{asin_mapping_error:,}"
    )

    print(
        f"ASIN missing from catalogue: "
        f"{missing_catalogue_asin:,}"
    )

    print(
        f"Duplicate candidate IDs:     "
        f"{duplicate_candidate_ids:,}"
    )

    print(
        f"Duplicate ASIN sets:         "
        f"{duplicate_asins:,}"
    )

    print(
        f"Train-history overlap:       "
        f"{train_history_overlap:,}"
    )

    # ======================================================
    # Success Rate
    # ======================================================

    if raw_records:

        success_rate = (
            len(valid_records)
            / len(raw_records)
        ) * 100

        print()

        print(
            f"Validated success rate:      "
            f"{success_rate:.2f}%"
        )

    print()
    print(
        f"Validated output:\n"
        f"{output_file}"
    )

    if invalid_records:

        print()

        print(
            f"Invalid-record report:\n"
            f"{invalid_output_file}"
        )

    # ======================================================
    # Example
    # ======================================================

    if valid_records:

        print()
        print("=" * 70)

        print(
            "EXAMPLE VALIDATED AUGMENTATION"
        )

        print("=" * 70)

        print(
            json.dumps(
                valid_records[0],
                indent=2,
                ensure_ascii=False
            )
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
        "11_build_augmented_interactions.py "
        f"--dataset {dataset_key}"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    validate_augmentations()