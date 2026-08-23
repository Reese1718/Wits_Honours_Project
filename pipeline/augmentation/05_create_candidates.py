"""
05_create_candidates.py


    python pipeline/augmentation/05_create_candidates.py --dataset fashion

    python pipeline/augmentation/05_create_candidates.py --dataset video_games

    python pipeline/augmentation/05_create_candidates.py --dataset movies_tv

"""

import json
import os
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

TOP_K_CANDIDATES = 50


# ==========================================================
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Create TF-IDF candidate sets "
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

    if value is None:
        return ""

    return str(value).strip()


def join_list(values):

    if not isinstance(
        values,
        list
    ):
        return ""

    return " ".join(
        clean_text(value)
        for value in values
        if clean_text(value)
    )


def build_product_text(product):
    """
    Build one textual representation of a catalogue product
    for TF-IDF similarity.
    """

    parts = []

    title = clean_text(
        product.get(
            "title"
        )
    )

    store = clean_text(
        product.get(
            "store"
        )
    )

    department = clean_text(
        product.get(
            "department"
        )
    )

    manufacturer = clean_text(
        product.get(
            "manufacturer"
        )
    )

    features = join_list(
        product.get(
            "features",
            []
        )
    )

    description = clean_text(
        product.get(
            "description"
        )
    )

    if title:
        parts.append(
            title
        )

    if store:
        parts.append(
            store
        )

    if department:
        parts.append(
            department
        )

    if manufacturer:
        parts.append(
            manufacturer
        )

    if features:
        parts.append(
            features
        )

    # Keep description shorter so it does not dominate
    if description:

        parts.append(
            description[:500]
        )

    return " ".join(
        parts
    )


def build_user_profile(history):
    """
    Combine the user's historical product metadata into
    one textual user-profile representation.
    """

    history_texts = []

    for product in history:

        text = build_product_text(
            product
        )

        if text:

            history_texts.append(
                text
            )

    return " ".join(
        history_texts
    )


# ==========================================================
# Main Function
# ==========================================================

def create_candidates():

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

    llm_input_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_inputs.json"
    )

    catalogue_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_catalogue.json"
    )

    output_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_candidates.json"
    )

    # ======================================================
    # Validate required inputs
    # ======================================================

    if not os.path.exists(
        llm_input_file
    ):

        raise FileNotFoundError(
            "\nLLM input file not found:\n"
            f"{llm_input_file}\n\n"
            "Run 04_create_llm_inputs.py "
            f"--dataset {dataset_key} first."
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

    # ======================================================
    # Header
    # ======================================================

    print(
        "=" * 70
    )

    print(
        f"CREATING "
        f"{dataset_folder.upper()} "
        f"LLM CANDIDATES"
    )

    print(
        "=" * 70
    )

    # ======================================================
    # Load catalogue
    # ======================================================

    print(
        "\nLoading LLM catalogue:"
    )

    print(
        catalogue_file
    )

    with open(
        catalogue_file,
        "r",
        encoding="utf-8"
    ) as file:

        catalogue = json.load(
            file
        )

    print(
        f"Catalogue products: "
        f"{len(catalogue):,}"
    )

    # ------------------------------------------------------
    # Build product texts
    # ------------------------------------------------------

    catalogue_texts = []
    catalogue_asins = []

    skipped_catalogue_products = 0

    for product in catalogue:

        asin = clean_text(
            product.get(
                "asin"
            )
        )

        if not asin:

            skipped_catalogue_products += 1
            continue

        text = build_product_text(
            product
        )

        # TF-IDF cannot work usefully with a completely
        # empty product representation.
        if not text:

            skipped_catalogue_products += 1
            continue

        catalogue_asins.append(
            asin
        )

        catalogue_texts.append(
            text
        )

    print(
        f"Usable catalogue products: "
        f"{len(catalogue_asins):,}"
    )

    print(
        f"Skipped catalogue products: "
        f"{skipped_catalogue_products:,}"
    )

    if not catalogue_texts:

        raise RuntimeError(
            "No usable catalogue product text "
            "was created."
        )

    # ------------------------------------------------------
    # ASIN -> catalogue product lookup
    # ------------------------------------------------------

    catalogue_lookup = {
        clean_text(
            product.get(
                "asin"
            )
        ): product

        for product in catalogue

        if clean_text(
            product.get(
                "asin"
            )
        )
    }

    # ======================================================
    # Build TF-IDF catalogue matrix
    # ======================================================

    print()
    print(
        "Building TF-IDF catalogue representation..."
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000
    )

    catalogue_matrix = (
        vectorizer.fit_transform(
            catalogue_texts
        )
    )

    print(
        "TF-IDF matrix shape:",
        catalogue_matrix.shape
    )

    # ======================================================
    # Load LLM user inputs
    # ======================================================

    print()
    print(
        "Loading LLM user inputs:"
    )

    print(
        llm_input_file
    )

    with open(
        llm_input_file,
        "r",
        encoding="utf-8"
    ) as file:

        llm_inputs = json.load(
            file
        )

    print(
        f"Users loaded: "
        f"{len(llm_inputs):,}"
    )

    # ======================================================
    # Generate Candidates
    # ======================================================

    output_records = []

    skipped_users = 0
    users_with_fewer_than_k = 0

    candidate_counts = []

    for user_number, user_record in enumerate(
        llm_inputs,
        start=1
    ):

        user_id = clean_text(
            user_record.get(
                "user_id"
            )
        )

        history = user_record.get(
            "history",
            []
        )

        if not user_id or not history:

            skipped_users += 1
            continue

        # --------------------------------------------------
        # User history ASINs
        # --------------------------------------------------

        history_asins = {
            clean_text(
                product.get(
                    "asin"
                )
            )

            for product in history

            if clean_text(
                product.get(
                    "asin"
                )
            )
        }

        # --------------------------------------------------
        # Build user text profile
        # --------------------------------------------------

        user_profile = build_user_profile(
            history
        )

        if not user_profile:

            skipped_users += 1
            continue

        # --------------------------------------------------
        # Transform user profile using catalogue vocabulary
        # --------------------------------------------------

        user_vector = vectorizer.transform(
            [
                user_profile
            ]
        )

        # --------------------------------------------------
        # Compute similarity to all catalogue items
        # --------------------------------------------------

        similarities = cosine_similarity(
            user_vector,
            catalogue_matrix
        ).flatten()

        # --------------------------------------------------
        # Rank products
        # --------------------------------------------------

        ranked_indices = (
            similarities
            .argsort()[
                ::-1
            ]
        )

        candidates = []

        candidate_number = 1

        for index in ranked_indices:

            asin = (
                catalogue_asins[
                    index
                ]
            )

            # Never recommend something already
            # present in the user's current history.
            if asin in history_asins:
                continue

            product = (
                catalogue_lookup[
                    asin
                ]
            )

            candidate_id = (
                f"C{candidate_number:03d}"
            )

            candidates.append(
                {
                    "candidate_id": (
                        candidate_id
                    ),

                    "asin": asin,

                    "similarity_score": float(
                        similarities[
                            index
                        ]
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

                    "features": product.get(
                        "features",
                        []
                    ),
                }
            )

            candidate_number += 1

            if (
                len(candidates)
                >= TOP_K_CANDIDATES
            ):

                break

        # --------------------------------------------------
        # Candidate checks
        # --------------------------------------------------

        candidate_count = len(
            candidates
        )

        candidate_counts.append(
            candidate_count
        )

        if (
            candidate_count
            < TOP_K_CANDIDATES
        ):

            users_with_fewer_than_k += 1

        # Verify no history ASIN appears in candidates
        candidate_asins = {
            candidate[
                "asin"
            ]
            for candidate
            in candidates
        }

        overlap = (
            candidate_asins
            & history_asins
        )

        if overlap:

            raise RuntimeError(
                "\nHISTORY/CANDIDATE OVERLAP DETECTED\n"
                f"User: {user_id}\n"
                f"ASINs: {sorted(overlap)}"
            )

        # --------------------------------------------------
        # Save user candidate set
        # --------------------------------------------------

        output_records.append(
            {
                "user_id": (
                    user_id
                ),

                "history_length": len(
                    history
                ),

                "history_asins": sorted(
                    history_asins
                ),

                "candidate_count": (
                    candidate_count
                ),

                "candidates": (
                    candidates
                ),
            }
        )

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (
            user_number
            % 250
            == 0
        ):

            print(
                f"Processed "
                f"{user_number:,} / "
                f"{len(llm_inputs):,} users"
            )

    # ======================================================
    # Global Validation
    # ======================================================

    print()
    print(
        "=" * 70
    )

    print(
        "VALIDATING CANDIDATE SETS"
    )

    print(
        "=" * 70
    )

    if not output_records:

        raise RuntimeError(
            "No candidate records were created."
        )

    user_ids = [
        record[
            "user_id"
        ]
        for record in output_records
    ]

    if (
        len(user_ids)
        != len(
            set(user_ids)
        )
    ):

        raise RuntimeError(
            "Duplicate user IDs detected "
            "in candidate output."
        )

    duplicate_candidate_users = 0

    for record in output_records:

        candidate_asins = [
            candidate[
                "asin"
            ]

            for candidate
            in record[
                "candidates"
            ]
        ]

        if (
            len(candidate_asins)
            != len(
                set(candidate_asins)
            )
        ):

            duplicate_candidate_users += 1

    if duplicate_candidate_users:

        raise RuntimeError(
            f"Duplicate candidate ASINs detected "
            f"for {duplicate_candidate_users} users."
        )

    print(
        "Candidate validation passed."
    )

    print(
        "No history ASINs appear "
        "in candidate sets."
    )

    print(
        "No duplicate users detected."
    )

    print(
        "No duplicate candidate ASINs "
        "within user candidate sets."
    )

    # ======================================================
    # Save Output
    # ======================================================

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_records,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ======================================================
    # Summary
    # ======================================================

    print()
    print(
        "=" * 70
    )

    print(
        f"{dataset_folder.upper()} "
        f"CANDIDATES CREATED"
    )

    print(
        "=" * 70
    )

    print(
        f"Dataset:                  "
        f"{dataset_folder}"
    )

    print(
        f"Users written:            "
        f"{len(output_records):,}"
    )

    print(
        f"Users skipped:            "
        f"{skipped_users:,}"
    )

    print(
        f"Target candidates/user:   "
        f"{TOP_K_CANDIDATES}"
    )

    print(
        f"Users with <50 candidates:"
        f" {users_with_fewer_than_k:,}"
    )

    if candidate_counts:

        print(
            f"Minimum candidate count:  "
            f"{min(candidate_counts):,}"
        )

        print(
            f"Maximum candidate count:  "
            f"{max(candidate_counts):,}"
        )

        print(
            f"Average candidate count:  "
            f"{sum(candidate_counts) / len(candidate_counts):.2f}"
        )

    print()
    print(
        f"Output file:\n"
        f"{output_file}"
    )

    # ======================================================
    # Example
    # ======================================================

    if output_records:

        print()
        print(
            "Example candidate set:"
        )

        print(
            "-" * 70
        )

        example = (
            output_records[
                0
            ]
        )

        print(
            json.dumps(
                {
                    "user_id": (
                        example[
                            "user_id"
                        ]
                    ),

                    "history_asins": (
                        example[
                            "history_asins"
                        ]
                    ),

                    "candidate_count": (
                        example[
                            "candidate_count"
                        ]
                    ),

                    "candidates": (
                        example[
                            "candidates"
                        ][:5]
                    ),
                },
                indent=2,
                ensure_ascii=False
            )
        )

    print()
    print(
        "=" * 70
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

    create_candidates()