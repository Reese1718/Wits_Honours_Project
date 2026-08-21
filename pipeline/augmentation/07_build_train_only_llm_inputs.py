"""
07_build_train_only_llm_inputs.py

Creates leakage-safe LLM inputs for Fashion.

Uses the existing chronological user histories produced by:
    03_build_user_histories.py

Reproduces the same leave-one-out split used by RecBole:

    eval_args:
        group_by: user
        order: TO
        split:
            LS: valid_and_test

For each user:

    train      = history[:-2]
    validation = history[-2]
    test       = history[-1]

Only the TRAIN portion is written to the LLM input file.

Output:
    Fashion_llm_train_inputs.json
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

USER_HISTORIES_FILE = os.path.join(
    PROJECT_DIR,
    "datasets",
    "Fashion",
    "exports",
    "Fashion_user_histories.json"
)

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "datasets",
    "Fashion",
    "exports",
    "Fashion_llm_train_inputs.json"
)


# ==========================================================
# Helper Functions
# ==========================================================

def clean_text(value):
    """
    Convert a value into a clean string.
    """

    if value is None:
        return ""

    return str(value).strip()


def compact_features(features, max_features=3):
    """
    Keep only a small number of product features.

    This matches the representation used by:
        04_create_fashion_llm_inputs.py
    """

    if not isinstance(features, list):
        return []

    cleaned = [
        clean_text(feature)
        for feature in features
        if clean_text(feature)
    ]

    return cleaned[:max_features]


def build_product_representation(product):
    """
    Create the same compact product representation used
    by 04_create_fashion_llm_inputs.py.
    """

    result = {
        "asin": clean_text(
            product.get("asin")
        ),
        "title": clean_text(
            product.get("title")
        ),
        "store": clean_text(
            product.get("store")
        ),
        "department": clean_text(
            product.get("department")
        ),
        "manufacturer": clean_text(
            product.get("manufacturer")
        )
    }

    features = compact_features(
        product.get("features", [])
    )

    if features:
        result["features"] = features

    description = clean_text(
        product.get("description")
    )

    useful_fields = [
        result.get("title"),
        result.get("store"),
        result.get("department"),
        result.get("manufacturer")
    ]

    useful_count = sum(
        1
        for value in useful_fields
        if value
    )

    if description and useful_count <= 2:

        result["description"] = (
            description[:500]
        )

    return result


# ==========================================================
# Build Train-Only Inputs
# ==========================================================

def build_train_only_inputs():

    print("=" * 70)
    print(
        "BUILDING TRAIN-ONLY FASHION LLM INPUTS"
    )
    print("=" * 70)

    # ------------------------------------------------------
    # Load existing user histories
    # ------------------------------------------------------

    print("\nLoading user histories:")
    print(USER_HISTORIES_FILE)

    if not os.path.exists(
        USER_HISTORIES_FILE
    ):

        raise FileNotFoundError(
            "\nUser history file not found:\n"
            f"{USER_HISTORIES_FILE}\n\n"
            "Run 03_build_user_histories.py first."
        )

    with open(
        USER_HISTORIES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        user_histories = json.load(file)

    print(
        f"Users loaded: {len(user_histories):,}"
    )

    # ------------------------------------------------------
    # Create train-only histories
    # ------------------------------------------------------

    llm_inputs = []

    skipped_users = 0

    total_full_interactions = 0
    total_train_interactions = 0

    validation_interactions = 0
    test_interactions = 0

    minimum_full_length = None
    maximum_full_length = 0

    minimum_train_length = None
    maximum_train_length = 0

    # Used for a structural validation at the end
    expected_train_sequences = {}

    for user_id, history in (
        user_histories.items()
    ):

        if not history:
            skipped_users += 1
            continue

        # --------------------------------------------------
        # Sort again defensively.
        #
        # File 03 already sorts chronologically, but doing it
        # again ensures this script is independently safe.
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

        full_length = len(history)

        total_full_interactions += (
            full_length
        )

        if minimum_full_length is None:
            minimum_full_length = (
                full_length
            )
        else:
            minimum_full_length = min(
                minimum_full_length,
                full_length
            )

        maximum_full_length = max(
            maximum_full_length,
            full_length
        )

        # --------------------------------------------------
        # RecBole leave-one-out requires:
        #
        # at least:
        #   1 train
        #   1 validation
        #   1 test
        # --------------------------------------------------

        if full_length < 3:

            skipped_users += 1
            continue

        # --------------------------------------------------
        # EXACT RECBOLE SPLIT
        # --------------------------------------------------

        train_history = history[:-2]

        validation_item = history[-2]

        test_item = history[-1]

        validation_interactions += 1
        test_interactions += 1

        # --------------------------------------------------
        # Compact ONLY the training history
        # --------------------------------------------------

        compact_history = []

        for product in train_history:

            product_representation = (
                build_product_representation(
                    product
                )
            )

            if not (
                product_representation[
                    "asin"
                ]
            ):
                continue

            compact_history.append(
                product_representation
            )

        if not compact_history:

            skipped_users += 1
            continue

        train_length = len(
            compact_history
        )

        total_train_interactions += (
            train_length
        )

        if minimum_train_length is None:

            minimum_train_length = (
                train_length
            )

        else:

            minimum_train_length = min(
                minimum_train_length,
                train_length
            )

        maximum_train_length = max(
            maximum_train_length,
            train_length
        )

        # --------------------------------------------------
        # Save expected ASIN sequence for validation.
        # --------------------------------------------------

        expected_train_sequences[
            user_id
        ] = [
            clean_text(
                product.get("asin")
            )
            for product in train_history
            if clean_text(
                product.get("asin")
            )
        ]

        # --------------------------------------------------
        # FINAL LLM RECORD
        #
        # Notice:
        # validation_item and test_item are NOT written.
        # --------------------------------------------------

        llm_inputs.append(
            {
                "user_id": user_id,

                "history_length": (
                    train_length
                ),

                "history": (
                    compact_history
                )
            }
        )

    # ======================================================
    # Structural Leakage Validation
    # ======================================================

    print()
    print("=" * 70)
    print("VALIDATING TRAIN-ONLY HISTORIES")
    print("=" * 70)

    validation_errors = 0

    for record in llm_inputs:

        user_id = record["user_id"]

        actual_sequence = [
            item["asin"]
            for item in record["history"]
        ]

        expected_sequence = (
            expected_train_sequences[
                user_id
            ]
        )

        if (
            actual_sequence
            != expected_sequence
        ):

            validation_errors += 1

            print(
                f"ERROR: Training sequence "
                f"mismatch for user "
                f"{user_id}"
            )

            if validation_errors >= 10:

                print(
                    "Stopping error display "
                    "after 10 mismatches."
                )

                break

    if validation_errors > 0:

        raise RuntimeError(
            "\nTRAIN-ONLY VALIDATION FAILED.\n"
            f"Mismatched users: "
            f"{validation_errors}"
        )

    print(
        "Validation passed."
    )

    print(
        "Every LLM history contains "
        "history[:-2] only."
    )

    print(
        "Validation and test positions "
        "are excluded."
    )

    # ======================================================
    # Save
    # ======================================================

    print()
    print("Saving output:")
    print(OUTPUT_FILE)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            llm_inputs,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ======================================================
    # Statistics
    # ======================================================

    print()
    print("=" * 70)
    print(
        "TRAIN-ONLY LLM INPUT DATASET CREATED"
    )
    print("=" * 70)

    print(
        f"Users written:              "
        f"{len(llm_inputs):,}"
    )

    print(
        f"Users skipped:              "
        f"{skipped_users:,}"
    )

    print(
        f"Full interactions:          "
        f"{total_full_interactions:,}"
    )

    print(
        f"Training interactions:      "
        f"{total_train_interactions:,}"
    )

    print(
        f"Validation interactions:    "
        f"{validation_interactions:,}"
    )

    print(
        f"Test interactions:          "
        f"{test_interactions:,}"
    )

    print(
        f"Shortest full history:      "
        f"{minimum_full_length}"
    )

    print(
        f"Longest full history:       "
        f"{maximum_full_length}"
    )

    print(
        f"Shortest train history:     "
        f"{minimum_train_length}"
    )

    print(
        f"Longest train history:      "
        f"{maximum_train_length}"
    )

    if llm_inputs:

        average_train_length = (
            total_train_interactions
            / len(llm_inputs)
        )

        print(
            f"Average train history:      "
            f"{average_train_length:.2f}"
        )

    # ======================================================
    # Example
    # ======================================================

    if llm_inputs:

        example = llm_inputs[0]

        example_user = (
            example["user_id"]
        )

        full_history = sorted(
            user_histories[
                example_user
            ],
            key=lambda item: (
                item.get("timestamp")
                if item.get("timestamp")
                is not None
                else 0
            )
        )

        print()
        print("=" * 70)
        print("EXAMPLE USER SPLIT")
        print("=" * 70)

        print(
            f"User ID:\n"
            f"{example_user}"
        )

        print()
        print(
            f"Full history length: "
            f"{len(full_history)}"
        )

        print(
            f"Train history length: "
            f"{len(full_history[:-2])}"
        )

        print()
        print("Train ASINs:")

        for product in (
            full_history[:-2]
        ):

            print(
                "  ",
                product.get("asin")
            )

        print()
        print(
            "Held-out validation ASIN:"
        )

        print(
            "  ",
            full_history[-2].get(
                "asin"
            )
        )

        print()
        print(
            "Held-out test ASIN:"
        )

        print(
            "  ",
            full_history[-1].get(
                "asin"
            )
        )

        print()
        print(
            "Actual LLM-visible record:"
        )

        print(
            json.dumps(
                example,
                indent=2,
                ensure_ascii=False
            )
        )

    print()
    print(
        f"Output file:\n{OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print()
    print(
        "Next step:"
    )

    print(
        "08_build_train_only_candidates.py"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    build_train_only_inputs()