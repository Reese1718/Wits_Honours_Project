"""
04_create_llm_inputs.py


    python pipeline/augmentation/04_create_llm_inputs.py --dataset fashion

    python pipeline/augmentation/04_create_llm_inputs.py --dataset video_games

    python pipeline/augmentation/04_create_llm_inputs.py --dataset movies_tv
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
            "Create compact LLM user inputs "
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
    Convert a value into a clean string.
    """

    if value is None:
        return ""

    return str(value).strip()


def compact_features(
    features,
    max_features=3
):
    """
    Keep only a small number of product features
    so LLM inputs do not become unnecessarily large.
    """

    if not isinstance(
        features,
        list
    ):
        return []

    cleaned = [
        clean_text(feature)
        for feature in features
        if clean_text(feature)
    ]

    return cleaned[
        :max_features
    ]


def build_product_representation(
    product
):
    """
    Create the compact product representation that will
    eventually be supplied to the LLM.

    ASIN is always retained because it is the canonical
    item identifier used by the sequential recommenders.
    """

    result = {
        "asin": clean_text(
            product.get(
                "asin"
            )
        ),

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
    }

    # ------------------------------------------------------
    # Add a few features if available
    # ------------------------------------------------------

    features = compact_features(
        product.get(
            "features",
            []
        )
    )

    if features:

        result[
            "features"
        ] = features

    # ------------------------------------------------------
    # Use description only if useful metadata is sparse
    # ------------------------------------------------------

    description = clean_text(
        product.get(
            "description"
        )
    )

    useful_fields = [
        result.get(
            "title"
        ),
        result.get(
            "store"
        ),
        result.get(
            "department"
        ),
        result.get(
            "manufacturer"
        ),
    ]

    useful_count = sum(
        1
        for value in useful_fields
        if value
    )

    # Only include description when other information
    # is limited.
    if (
        description
        and useful_count <= 2
    ):

        # Prevent extremely long descriptions
        # from exploding prompts.
        result[
            "description"
        ] = description[:500]

    return result


# ==========================================================
# Main Function
# ==========================================================

def create_llm_inputs():

    args = parse_args()

    dataset_key = (
        args.dataset
    )

    config = (
        DATASETS[
            dataset_key
        ]
    )

    dataset_folder = (
        config[
            "folder"
        ]
    )

    dataset_prefix = (
        config[
            "prefix"
        ]
    )

    # ======================================================
    # Project Paths
    # ======================================================

    script_dir = os.path.dirname(
        os.path.abspath(
            __file__
        )
    )

    project_dir = os.path.dirname(
        os.path.dirname(
            script_dir
        )
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

    user_histories_file = (
        os.path.join(
            export_dir,
            f"{dataset_prefix}_user_histories.json"
        )
    )

    output_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_inputs.json"
    )

    # ======================================================
    # Validate input
    # ======================================================

    if not os.path.exists(
        user_histories_file
    ):

        raise FileNotFoundError(
            "\nUser histories file not found:\n"
            f"{user_histories_file}\n\n"
            "Run 03_build_user_histories.py "
            f"--dataset {dataset_key} first."
        )

    # ======================================================
    # Header
    # ======================================================

    print(
        "=" * 70
    )

    print(
        f"CREATING "
        f"{dataset_folder.upper()} "
        f"LLM INPUTS"
    )

    print(
        "=" * 70
    )

    # ======================================================
    # Load user histories
    # ======================================================

    print(
        "\nLoading user histories:"
    )

    print(
        user_histories_file
    )

    with open(
        user_histories_file,
        "r",
        encoding="utf-8"
    ) as file:

        user_histories = (
            json.load(
                file
            )
        )

    print(
        f"Users loaded: "
        f"{len(user_histories):,}"
    )

    # ======================================================
    # Build LLM input records
    # ======================================================

    llm_inputs = []

    skipped_users = 0
    total_history_items = 0

    for (
        user_id,
        history
    ) in user_histories.items():

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
                item.get(
                    "timestamp"
                )
                if item.get(
                    "timestamp"
                ) is not None
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

        total_history_items += len(
            compact_history
        )

        llm_inputs.append(
            {
                "user_id": str(
                    user_id
                ),

                "history_length": len(
                    compact_history
                ),

                "history": (
                    compact_history
                ),
            }
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
            llm_inputs,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ======================================================
    # Statistics
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "LLM INPUT DATASET CREATED"
    )

    print(
        "=" * 70
    )

    print(
        f"Dataset:                "
        f"{dataset_folder}"
    )

    print(
        f"Users written:          "
        f"{len(llm_inputs):,}"
    )

    print(
        f"Users skipped:          "
        f"{skipped_users:,}"
    )

    print(
        f"History items written:  "
        f"{total_history_items:,}"
    )

    if llm_inputs:

        average_history_length = (
            total_history_items
            / len(
                llm_inputs
            )
        )

        print(
            f"Average history length: "
            f"{average_history_length:.2f}"
        )

    print(
        f"\nOutput file:\n"
        f"{output_file}"
    )

    # ======================================================
    # Example record
    # ======================================================

    if llm_inputs:

        print(
            "\nExample LLM input:"
        )

        print(
            "-" * 70
        )

        print(
            json.dumps(
                llm_inputs[0],
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

    create_llm_inputs()