"""
03_build_user_histories.py


    python pipeline/augmentation/03_build_user_histories.py --dataset fashion

    python pipeline/augmentation/03_build_user_histories.py --dataset video_games

    python pipeline/augmentation/03_build_user_histories.py --dataset movies_tv
"""

import json
import os
import argparse
from collections import defaultdict


# ==========================================================
# Dataset Configuration
# ==========================================================

DATASETS = {
    "fashion": {
        "folder": "Fashion",
        "prefix": "Fashion",
        "filtered_file": "Fashion_filtered.jsonl",
    },

    "video_games": {
        "folder": "Video Games",
        "prefix": "Video_Games",

        # Confirm this against your actual processed filename
        "filtered_file": "Video_Games_filtered.jsonl",
    },

    "movies_tv": {
        "folder": "Movies and TV Shows",
        "prefix": "Movies_TV",

        # Confirm this against your actual processed filename
        "filtered_file": "Movies_TV_filtered.jsonl",
    },
}


# ==========================================================
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Build chronological user histories "
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
# Main Function
# ==========================================================

def build_user_histories():

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

    processed_dir = os.path.join(
        dataset_dir,
        "processed"
    )

    export_dir = os.path.join(
        dataset_dir,
        "exports"
    )

    interaction_file = os.path.join(
        processed_dir,
        config["filtered_file"]
    )

    catalogue_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_catalogue.json"
    )

    output_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_user_histories.json"
    )

    # ======================================================
    # Validate inputs
    # ======================================================

    if not os.path.exists(
        interaction_file
    ):

        raise FileNotFoundError(
            "\nFiltered interaction file not found:\n"
            f"{interaction_file}"
        )

    if not os.path.exists(
        catalogue_file
    ):

        raise FileNotFoundError(
            "\nLLM catalogue file not found:\n"
            f"{catalogue_file}\n\n"
            "Run 02_build_llm_catalogue.py "
            f"--dataset {dataset_key} first."
        )

    os.makedirs(
        export_dir,
        exist_ok=True
    )

    # ======================================================
    # Header
    # ======================================================

    print("=" * 70)

    print(
        f"BUILDING "
        f"{dataset_folder.upper()} "
        f"USER HISTORIES"
    )

    print("=" * 70)

    # ======================================================
    # Load product catalogue
    # ======================================================

    print(
        "\nLoading LLM product catalogue:"
    )

    print(
        catalogue_file
    )

    with open(
        catalogue_file,
        "r",
        encoding="utf-8"
    ) as file:

        catalogue_list = json.load(file)

    # ------------------------------------------------------
    # Convert catalogue into ASIN lookup
    # ------------------------------------------------------

    catalogue = {
        str(product["asin"]): product
        for product in catalogue_list
        if product.get("asin")
    }

    print(
        f"Products in catalogue: "
        f"{len(catalogue):,}"
    )

    # ======================================================
    # Load filtered interactions
    # ======================================================

    print(
        "\nLoading filtered interaction dataset:"
    )

    print(
        interaction_file
    )

    user_histories = defaultdict(
        list
    )

    total_interactions = 0
    missing_metadata = 0
    skipped_invalid_interactions = 0

    with open(
        interaction_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            interaction = json.loads(
                line
            )

            user_id = interaction.get(
                "user_id"
            )

            item_id = interaction.get(
                "item_id"
            )

            timestamp = interaction.get(
                "timestamp"
            )

            if not user_id or not item_id:

                skipped_invalid_interactions += 1
                continue

            user_id = str(
                user_id
            )

            item_id = str(
                item_id
            )

            total_interactions += 1

            # ------------------------------------------------
            # Find product metadata using ASIN
            # ------------------------------------------------

            product = catalogue.get(
                item_id
            )

            if product is None:

                missing_metadata += 1
                continue

            # ------------------------------------------------
            # Add interaction to user's history
            # ------------------------------------------------

            user_histories[
                user_id
            ].append(
                {
                    "asin": item_id,

                    "timestamp": (
                        timestamp
                    ),

                    "title": product.get(
                        "title",
                        ""
                    ),

                    "store": product.get(
                        "store",
                        ""
                    ),

                    "main_category": product.get(
                        "main_category",
                        ""
                    ),

                    "categories": product.get(
                        "categories",
                        []
                    ),

                    "department": product.get(
                        "department",
                        ""
                    ),

                    "manufacturer": product.get(
                        "manufacturer",
                        ""
                    ),

                    "description": product.get(
                        "description",
                        ""
                    ),

                    "features": product.get(
                        "features",
                        []
                    ),
                }
            )

    # ======================================================
    # Sort histories chronologically
    # ======================================================

    for user_id in user_histories:

        user_histories[
            user_id
        ].sort(
            key=lambda x: (
                x["timestamp"]
                if x["timestamp"] is not None
                else 0
            )
        )

    # ======================================================
    # Convert to regular dictionary
    # ======================================================

    user_histories = dict(
        user_histories
    )

    # ======================================================
    # Save output
    # ======================================================

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            user_histories,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ======================================================
    # Statistics
    # ======================================================

    total_users = len(
        user_histories
    )

    history_lengths = [
        len(history)
        for history
        in user_histories.values()
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "USER HISTORY DATASET CREATED"
    )

    print(
        "=" * 70
    )

    print(
        f"Dataset:                "
        f"{dataset_folder}"
    )

    print(
        f"Users:                  "
        f"{total_users:,}"
    )

    print(
        f"Interactions scanned:   "
        f"{total_interactions:,}"
    )

    print(
        f"Users with histories:    "
        f"{total_users:,}"
    )

    print(
        f"Missing metadata:        "
        f"{missing_metadata:,}"
    )

    print(
        f"Invalid interactions:    "
        f"{skipped_invalid_interactions:,}"
    )

    if history_lengths:

        print(
            f"Average history length:  "
            f"{sum(history_lengths) / len(history_lengths):.2f}"
        )

        print(
            f"Shortest history:        "
            f"{min(history_lengths)}"
        )

        print(
            f"Longest history:         "
            f"{max(history_lengths)}"
        )

    print(
        f"\nOutput file:\n"
        f"{output_file}"
    )

    # ======================================================
    # Example user
    # ======================================================

    if user_histories:

        example_user = next(
            iter(user_histories)
        )

        print(
            "\nExample user:"
        )

        print(
            "-" * 70
        )

        print(
            "User ID:",
            example_user
        )

        print(
            json.dumps(
                user_histories[
                    example_user
                ],
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

    build_user_histories()