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

INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "datasets",
    "Fashion",
    "exports",
    "Fashion_metadata_lookup.json"
)

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "datasets",
    "Fashion",
    "exports",
    "Fashion_llm_catalogue.json"
)


# ==========================================================
# Helper Functions
# ==========================================================

def clean_text(value):
    """
    Convert a metadata field into a clean string.
    """

    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(
            str(item).strip()
            for item in value
            if item
        )

    return str(value).strip()


def clean_list(value):
    """
    Convert a metadata field into a clean list of strings.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if item
        ]

    value = str(value).strip()

    if not value:
        return []

    return [value]


# ==========================================================
# Main Function
# ==========================================================

def build_llm_catalogue():

    print("=" * 70)
    print("BUILDING FASHION LLM CATALOGUE")
    print("=" * 70)

    # ------------------------------------------------------
    # Load metadata lookup
    # ------------------------------------------------------

    print("\nLoading metadata lookup:")
    print(INPUT_FILE)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    print(
        f"Metadata records loaded: {len(metadata):,}"
    )

    # ------------------------------------------------------
    # Build catalogue
    # ------------------------------------------------------

    catalogue = []

    missing_title = 0
    missing_store = 0
    missing_main_category = 0
    missing_categories = 0
    missing_department = 0
    missing_manufacturer = 0
    missing_description = 0
    missing_features = 0

    # Fashion_metadata_lookup.json is keyed by ASIN
    for product in metadata.values():

        # --------------------------------------------------
        # Mandatory identifier
        # --------------------------------------------------

        asin = clean_text(
            product.get("asin")
        )

        if not asin:
            continue

        # --------------------------------------------------
        # Basic metadata
        # --------------------------------------------------

        title = clean_text(
            product.get("title")
        )

        store = clean_text(
            product.get("store")
        )

        main_category = clean_text(
            product.get("main_category")
        )

        categories = clean_list(
            product.get("categories")
        )

        description = clean_text(
            product.get("description")
        )

        features = clean_list(
            product.get("features")
        )

        # --------------------------------------------------
        # Extract useful values from details
        # --------------------------------------------------

        details = product.get(
            "details",
            {}
        )

        if not isinstance(details, dict):
            details = {}

        department = clean_text(
            details.get("Department")
        )

        manufacturer = clean_text(
            details.get("Manufacturer")
        )

        # --------------------------------------------------
        # Track missing metadata
        # --------------------------------------------------

        if not title:
            missing_title += 1

        if not store:
            missing_store += 1

        if not main_category:
            missing_main_category += 1

        if not categories:
            missing_categories += 1

        if not department:
            missing_department += 1

        if not manufacturer:
            missing_manufacturer += 1

        if not description:
            missing_description += 1

        if not features:
            missing_features += 1

        # --------------------------------------------------
        # Create LLM catalogue record
        # --------------------------------------------------

        catalogue.append(
            {
                "asin": asin,
                "title": title,
                "store": store,
                "main_category": main_category,
                "categories": categories,
                "department": department,
                "manufacturer": manufacturer,
                "description": description,
                "features": features
            }
        )

    # ------------------------------------------------------
    # Save catalogue
    # ------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            catalogue,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ======================================================
    # Summary
    # ======================================================

    print("\n" + "=" * 70)
    print("LLM CATALOGUE CREATED")
    print("=" * 70)

    print(
        f"Products in catalogue:      {len(catalogue):,}"
    )

    print(
        f"Missing titles:             {missing_title:,}"
    )

    print(
        f"Missing stores:             {missing_store:,}"
    )

    print(
        f"Missing main categories:    {missing_main_category:,}"
    )

    print(
        f"Missing category lists:     {missing_categories:,}"
    )

    print(
        f"Missing departments:        {missing_department:,}"
    )

    print(
        f"Missing manufacturers:      {missing_manufacturer:,}"
    )

    print(
        f"Missing descriptions:       {missing_description:,}"
    )

    print(
        f"Missing features:           {missing_features:,}"
    )

    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )

    # ======================================================
    # Example Product
    # ======================================================

    if catalogue:

        print("\nExample catalogue record:")
        print("-" * 70)

        print(
            json.dumps(
                catalogue[0],
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

    build_llm_catalogue()