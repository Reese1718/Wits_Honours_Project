"""
10_validate_fashion_augmentations.py

Validates the raw Fashion LLM augmentation output created by:

    09_generate_fashion_augmentations.py

Checks that every accepted augmentation:

    1. Belongs to a valid user.
    2. Contains exactly 3 selected items.
    3. Uses valid candidate IDs.
    4. Maps each candidate ID to the correct candidate ASIN.
    5. Uses ASINs that exist in the Fashion LLM catalogue.
    6. Does not contain duplicate candidate IDs.
    7. Does not contain duplicate ASINs.
    8. Does not recommend an item already in the user's TRAIN history.

Produces:

    Fashion_llm_augmentations.json

This clean file is the one that should be used by step 11.
"""

import json
import os


# ==========================================================
# Project Paths
# ==========================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    os.path.dirname(SCRIPT_DIR)
)

EXPORT_DIR = os.path.join(
    PROJECT_DIR,
    "datasets",
    "Fashion",
    "exports"
)

RAW_AUGMENTATION_FILE = os.path.join(
    EXPORT_DIR,
    "Fashion_llm_augmentations_raw.json"
)

TRAIN_INPUT_FILE = os.path.join(
    EXPORT_DIR,
    "Fashion_llm_train_inputs.json"
)

TRAIN_CANDIDATE_FILE = os.path.join(
    EXPORT_DIR,
    "Fashion_llm_train_candidates.json"
)

CATALOGUE_FILE = os.path.join(
    EXPORT_DIR,
    "Fashion_llm_catalogue.json"
)

OUTPUT_FILE = os.path.join(
    EXPORT_DIR,
    "Fashion_llm_augmentations.json"
)


# ==========================================================
# Configuration
# ==========================================================

EXPECTED_AUGMENTATIONS = 3


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

    print("=" * 70)
    print(
        "VALIDATING FASHION LLM AUGMENTATIONS"
    )
    print("=" * 70)

    # ======================================================
    # Load Files
    # ======================================================

    print()
    raw_records = load_json(
        RAW_AUGMENTATION_FILE
    )

    print(
        f"Raw augmentation records: "
        f"{len(raw_records):,}"
    )

    print()
    train_inputs = load_json(
        TRAIN_INPUT_FILE
    )

    print(
        f"Train-only input users: "
        f"{len(train_inputs):,}"
    )

    print()
    candidate_records = load_json(
        TRAIN_CANDIDATE_FILE
    )

    print(
        f"Candidate records: "
        f"{len(candidate_records):,}"
    )

    print()
    catalogue = load_json(
        CATALOGUE_FILE
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

        selected_candidates = (
            raw_record.get(
                "selected_candidates",
                []
            )
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

        train_history = (
            train_record.get(
                "history",
                []
            )
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

            # Candidate ID must exist
            candidate = (
                candidate_id_lookup.get(
                    candidate_id
                )
            )

            if candidate is None:

                errors.append(
                    f"Invalid candidate ID: "
                    f"{candidate_id}"
                )

                invalid_candidate_id += 1

                continue

            expected_asin = clean_text(
                candidate.get("asin")
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
                    "asin": asin,
                    "title": clean_text(
                        candidate.get(
                            "title"
                        )
                    ),
                    "similarity_score": (
                        candidate.get(
                            "similarity_score"
                        )
                    )
                }
            )

        # --------------------------------------------------
        # Final record decision
        # --------------------------------------------------

        if errors:

            invalid_records.append(
                {
                    "user_id": user_id,
                    "errors": errors,
                    "raw_model_response": (
                        raw_record.get(
                            "raw_model_response",
                            ""
                        )
                    )
                }
            )

            continue

        if (
            len(clean_selected_items)
            != EXPECTED_AUGMENTATIONS
        ):

            invalid_records.append(
                {
                    "user_id": user_id,
                    "errors": [
                        "Final validated item count "
                        "is not exactly "
                        f"{EXPECTED_AUGMENTATIONS}"
                    ]
                }
            )

            continue

        # --------------------------------------------------
        # Clean final augmentation
        # --------------------------------------------------

        valid_records.append(
            {
                "user_id": user_id,

                "augmented_asins": [
                    item["asin"]
                    for item
                    in clean_selected_items
                ],

                "selected_candidates": (
                    clean_selected_items
                )
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
    # --------------------------------------------------

    valid_user_ids = [
        record["user_id"]
        for record in valid_records
    ]

    if (
        len(valid_user_ids)
        != len(set(valid_user_ids))
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
    # --------------------------------------------------

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
                "contain exactly 3 ASINs."
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
    print("Saving validated augmentations:")
    print(OUTPUT_FILE)

    with open(
        OUTPUT_FILE,
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
    # Optional invalid report
    # ======================================================

    INVALID_OUTPUT_FILE = os.path.join(
        EXPORT_DIR,
        "Fashion_llm_augmentations_invalid.json"
    )

    if invalid_records:

        with open(
            INVALID_OUTPUT_FILE,
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
        "FASHION AUGMENTATION VALIDATION COMPLETE"
    )
    print("=" * 70)

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
    print("Failure breakdown")
    print("-" * 70)

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
        f"{OUTPUT_FILE}"
    )

    if invalid_records:

        print()
        print(
            f"Invalid-record report:\n"
            f"{INVALID_OUTPUT_FILE}"
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
    print("COMPLETE")
    print("=" * 70)

    print()
    print("Next step:")
    print(
        "11_build_augmented_interactions.py"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    validate_augmentations()
