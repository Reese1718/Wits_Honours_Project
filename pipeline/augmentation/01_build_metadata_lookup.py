"""
01_build_metadata_lookup.py


    python pipeline/augmentation/01_build_metadata_lookup.py --dataset fashion

    python pipeline/augmentation/01_build_metadata_lookup.py --dataset video_games

    python pipeline/augmentation/01_build_metadata_lookup.py --dataset movies_tv
"""

import os
import json
import argparse


# ==========================================================
# Dataset Configuration
# ==========================================================

DATASETS = {
    "fashion": {
        "folder": "Fashion",
        "prefix": "Fashion",
        "filtered_file": "Fashion_filtered.jsonl",
        "metadata_file": "meta_Amazon_Fashion.jsonl",
    },

    "video_games": {
        "folder": "Video Games",
        "prefix": "Video_Games",
        "filtered_file": "Video_Games_filtered.jsonl",

        # CHECK THIS FILENAME against your actual dataset
        "metadata_file": "meta_Video_Games.jsonl",
    },

    "movies_tv": {
        "folder": "Movies and TV Shows",
        "prefix": "Movies_TV",
        "filtered_file": "Movies_TV_filtered.jsonl",

        # CHECK THIS FILENAME against your actual dataset
        "metadata_file": "meta_Movies_and_TV.jsonl",
    },
}


# ==========================================================
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build the metadata lookup for an Amazon dataset."
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
# Main
# ==========================================================

def main():

    args = parse_args()

    dataset_key = args.dataset
    config = DATASETS[dataset_key]

    dataset_folder = config["folder"]
    dataset_prefix = config["prefix"]

    # ======================================================
    # Paths
    # ======================================================

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    pipeline_dir = os.path.dirname(
        script_dir
    )

    project_dir = os.path.dirname(
        pipeline_dir
    )

    dataset_dir = os.path.join(
        project_dir,
        "datasets",
        dataset_folder
    )

    processed_dir = os.path.join(
        dataset_dir,
        "processed"
    )

    export_dir = os.path.join(
        dataset_dir,
        "exports"
    )

    filtered_file = os.path.join(
        processed_dir,
        config["filtered_file"]
    )

    metadata_file = os.path.join(
        dataset_dir,
        config["metadata_file"]
    )

    output_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_metadata_lookup.json"
    )

    # ======================================================
    # Validate inputs
    # ======================================================

    if not os.path.exists(
        filtered_file
    ):

        raise FileNotFoundError(
            "\nFiltered interaction file not found:\n"
            f"{filtered_file}"
        )

    if not os.path.exists(
        metadata_file
    ):

        raise FileNotFoundError(
            "\nMetadata file not found:\n"
            f"{metadata_file}"
        )

    # ======================================================
    # Create output directory
    # ======================================================

    os.makedirs(
        export_dir,
        exist_ok=True
    )

    # ======================================================
    # Load filtered ASINs
    # ======================================================

    print("=" * 70)

    print(
        f"BUILDING {dataset_folder.upper()} "
        f"METADATA LOOKUP"
    )

    print("=" * 70)

    print(
        "\nLoading filtered interaction dataset:"
    )

    print(
        filtered_file
    )

    filtered_asins = set()

    with open(
        filtered_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            if "item_id" in record:

                asin = record[
                    "item_id"
                ]

            elif "asin" in record:

                asin = record[
                    "asin"
                ]

            elif "parent_asin" in record:

                asin = record[
                    "parent_asin"
                ]

            else:
                continue

            filtered_asins.add(
                str(asin)
            )

    print(
        "\nUnique filtered ASINs:",
        len(filtered_asins)
    )

    # ======================================================
    # Load metadata
    # ======================================================

    print(
        f"\nLoading {dataset_folder} metadata:"
    )

    print(
        metadata_file
    )

    metadata_lookup = {}

    metadata_records = 0
    matched_records = 0

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            metadata_records += 1

            asin = (
                record.get(
                    "parent_asin"
                )
                or record.get(
                    "asin"
                )
            )

            if asin is None:
                continue

            asin = str(
                asin
            )

            # Only retain products in the filtered
            # interaction dataset
            if asin not in filtered_asins:
                continue

            metadata_lookup[
                asin
            ] = {
                "asin": asin,

                "title": record.get(
                    "title"
                ),

                "main_category": record.get(
                    "main_category"
                ),

                "categories": record.get(
                    "categories"
                ),

                "store": record.get(
                    "store"
                ),

                "details": record.get(
                    "details"
                ),

                "description": record.get(
                    "description"
                ),

                "features": record.get(
                    "features"
                ),
            }

            matched_records += 1

    # ======================================================
    # Save lookup
    # ======================================================

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata_lookup,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ======================================================
    # Validation
    # ======================================================

    matched_asins = set(
        metadata_lookup.keys()
    )

    missing_asins = (
        filtered_asins
        - matched_asins
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "METADATA LOOKUP COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nDataset:",
        dataset_folder
    )

    print(
        "Metadata records scanned:",
        metadata_records
    )

    print(
        "Filtered ASINs:",
        len(filtered_asins)
    )

    print(
        "Matched ASINs:",
        len(matched_asins)
    )

    print(
        "Missing ASINs:",
        len(missing_asins)
    )

    print(
        "\nOutput:",
        output_file
    )

    # ======================================================
    # Show examples
    # ======================================================

    print(
        "\nExample metadata records:"
    )

    print(
        "-" * 70
    )

    for asin in list(
        metadata_lookup.keys()
    )[:5]:

        print(
            json.dumps(
                metadata_lookup[
                    asin
                ],
                indent=2,
                ensure_ascii=False
            )
        )

    # ======================================================
    # Show missing examples
    # ======================================================

    if missing_asins:

        print(
            "\nExample missing ASINs:"
        )

        print(
            "-" * 70
        )

        for asin in list(
            missing_asins
        )[:10]:

            print(
                asin
            )

    else:

        print(
            "\nAll filtered ASINs have "
            "corresponding metadata."
        )

    print(
        "\n" + "=" * 70
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()