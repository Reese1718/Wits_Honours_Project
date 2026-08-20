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
    "Fashion_llm_inputs.json"
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
    Keep only a small number of product features
    so LLM inputs do not become unnecessarily large.
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
    Create the compact product representation that will
    eventually be supplied to the LLM.

    ASIN is always retained because it is the canonical
    item identifier used by the sequential recommenders.
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

    # ------------------------------------------------------
    # Add a few features if available
    # ------------------------------------------------------

    features = compact_features(
        product.get("features", [])
    )

    if features:
        result["features"] = features

    # ------------------------------------------------------
    # Use description only if useful metadata is sparse
    # ------------------------------------------------------

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

    # Only include description when other information is limited
    if description and useful_count <= 2:

        # Prevent extremely long descriptions from exploding prompts
        result["description"] = description[:500]

    return result


# ==========================================================
# Main Function
# ==========================================================

def create_llm_inputs():

    print("=" * 70)
    print("CREATING FASHION LLM INPUTS")
    print("=" * 70)

    # ======================================================
    # Load user histories
    # ======================================================

    print("\nLoading user histories:")
    print(USER_HISTORIES_FILE)

    with open(
        USER_HISTORIES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        user_histories = json.load(file)

    print(
        f"Users loaded: {len(user_histories):,}"
    )

    # ======================================================
    # Build LLM input records
    # ======================================================

    llm_inputs = []

    skipped_users = 0
    total_history_items = 0

    for user_id, history in user_histories.items():

        if not history:
            skipped_users += 1
            continue

        # --------------------------------------------------
        # Histories should already be chronological,
        # but sort again for safety.
        # --------------------------------------------------

        history = sorted(
            history,
            key=lambda item: (
                item.get("timestamp")
                if item.get("timestamp") is not None
                else 0
            )
        )

        compact_history = []

        for product in history:

            product_representation = (
                build_product_representation(
                    product
                )
            )

            # ASIN is mandatory
            if not product_representation["asin"]:
                continue

            compact_history.append(
                product_representation
            )

        if not compact_history:
            skipped_users += 1
            continue

        total_history_items += len(
            compact_history
        )

        llm_inputs.append(
            {
                "user_id": user_id,
                "history_length": len(
                    compact_history
                ),
                "history": compact_history
            }
        )

    # ======================================================
    # Save output
    # ======================================================

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

    print("\n" + "=" * 70)
    print("LLM INPUT DATASET CREATED")
    print("=" * 70)

    print(
        f"Users written:          {len(llm_inputs):,}"
    )

    print(
        f"Users skipped:          {skipped_users:,}"
    )

    print(
        f"History items written:  {total_history_items:,}"
    )

    if llm_inputs:

        average_history_length = (
            total_history_items
            / len(llm_inputs)
        )

        print(
            f"Average history length: "
            f"{average_history_length:.2f}"
        )

    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )

    # ======================================================
    # Example record
    # ======================================================

    if llm_inputs:

        print("\nExample LLM input:")
        print("-" * 70)

        print(
            json.dumps(
                llm_inputs[0],
                indent=2,
                ensure_ascii=False
            )
        )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    create_llm_inputs()