"""
02_build_llm_catalogue.py

    python pipeline/augmentation/02_build_llm_catalogue.py --dataset fashion

    python pipeline/augmentation/02_build_llm_catalogue.py --dataset video_games

    python pipeline/augmentation/02_build_llm_catalogue.py --dataset movies_tv
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
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build the cleaned LLM catalogue "
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

    dataset_dir = os.path.join(
        project_dir,
        "datasets",
        dataset_folder
    )

    export_dir = os.path.join(
        dataset_dir,
        "exports"
    )

    input_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_metadata_lookup.json"
    )

    output_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_catalogue.json"
    )

    # ======================================================
    # Validate Input
    # ======================================================

    if not os.path.exists(
        input_file
    ):

        raise FileNotFoundError(
            "\nMetadata lookup not found:\n"
            f"{input_file}\n\n"
            "Run 01_build_metadata_lookup.py "
            f"--dataset {dataset_key} first."
        )

    # ======================================================
    # Header
    # ======================================================

    print("=" * 70)

    print(
        f"BUILDING "
        f"{dataset_folder.upper()} "
        f"LLM CATALOGUE"
    )

    print("=" * 70)

    # ======================================================
    # Load metadata lookup
    # ======================================================

    print(
        "\nLoading metadata lookup:"
    )

    print(
        input_file
    )

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    print(
        f"Metadata records loaded: "
        f"{len(metadata):,}"
    )

    # ======================================================
    # Build catalogue
    # ======================================================

    catalogue = []

    missing_title = 0
    missing_store = 0
    missing_main_category = 0
    missing_categories = 0
    missing_department = 0
    missing_manufacturer = 0
    missing_description = 0
    missing_features = 0

    # Metadata lookup JSON is keyed by ASIN
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

        if not isinstance(
            details,
            dict
        ):
            details = {}

        department = clean_text(
            details.get(
                "Department"
            )
        )

        manufacturer = clean_text(
            details.get(
                "Manufacturer"
            )
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
                "features": features,
            }
        )

    # ======================================================
    # Save catalogue
    # ======================================================

    with open(
        output_file,
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

    print(
        "\n" + "=" * 70
    )

    print(
        "LLM CATALOGUE CREATED"
    )

    print(
        "=" * 70
    )

    print(
        f"Dataset:                    "
        f"{dataset_folder}"
    )

    print(
        f"Products in catalogue:      "
        f"{len(catalogue):,}"
    )

    print(
        f"Missing titles:             "
        f"{missing_title:,}"
    )

    print(
        f"Missing stores:             "
        f"{missing_store:,}"
    )

    print(
        f"Missing main categories:    "
        f"{missing_main_category:,}"
    )

    print(
        f"Missing category lists:     "
        f"{missing_categories:,}"
    )

    print(
        f"Missing departments:        "
        f"{missing_department:,}"
    )

    print(
        f"Missing manufacturers:      "
        f"{missing_manufacturer:,}"
    )

    print(
        f"Missing descriptions:       "
        f"{missing_description:,}"
    )

    print(
        f"Missing features:           "
        f"{missing_features:,}"
    )

    print(
        f"\nOutput file:\n"
        f"{output_file}"
    )

    # ======================================================
    # Example Product
    # ======================================================

    if catalogue:

        print(
            "\nExample catalogue record:"
        )

        print(
            "-" * 70
        )

        print(
            json.dumps(
                catalogue[0],
                indent=2,
                ensure_ascii=False
            )
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "COMPLETE"
    )

    print(
        "=" * 70
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    build_llm_catalogue()