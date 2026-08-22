"""
08_build_train_only_candidates.py

Builds 50 candidate products per Fashion user using ONLY the
train-only LLM histories created by:

    07_build_train_only_llm_inputs.py

Candidate generation methodology is intentionally kept the same
as the original:

    05_create_fashion_candidates.py

Method:
    1. Build TF-IDF representations of the Fashion catalogue.
    2. Build a textual profile from each user's TRAIN history.
    3. Compute cosine similarity between the user profile and
       every catalogue product.
    4. Exclude products already present in the TRAIN history.
    5. Keep the top 50 remaining products.

Output:
    Fashion_llm_train_candidates.json
"""

import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# Project Paths
# ==========================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    os.path.dirname(SCRIPT_DIR)
)

LLM_INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "datasets",
    "Fashion",
    "exports",
    "Fashion_llm_train_inputs.json"
)

CATALOGUE_FILE = os.path.join(
    PROJECT_DIR,
    "datasets",
    "Fashion",
    "exports",
    "Fashion_llm_catalogue.json"
)

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "datasets",
    "Fashion",
    "exports",
    "Fashion_llm_train_candidates.json"
)


# ==========================================================
# Configuration
# ==========================================================

TOP_K_CANDIDATES = 50


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


def join_list(values):
    """
    Convert a list of text fields into a single string.
    """

    if not isinstance(values, list):
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

    This matches the original candidate-generation logic
    used in 05_create_fashion_candidates.py.
    """

    parts = []

    title = clean_text(
        product.get("title")
    )

    store = clean_text(
        product.get("store")
    )

    department = clean_text(
        product.get("department")
    )

    manufacturer = clean_text(
        product.get("manufacturer")
    )

    features = join_list(
        product.get("features", [])
    )

    description = clean_text(
        product.get("description")
    )

    if title:
        parts.append(title)

    if store:
        parts.append(store)

    if department:
        parts.append(department)

    if manufacturer:
        parts.append(manufacturer)

    if features:
        parts.append(features)

    # Keep description shorter so it does not dominate
    if description:
        parts.append(
            description[:500]
        )

    return " ".join(parts)


def build_user_profile(history):
    """
    Combine the user's TRAIN-history product metadata into
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

def create_train_only_candidates():

    print("=" * 70)
    print(
        "BUILDING TRAIN-ONLY FASHION LLM CANDIDATES"
    )
    print("=" * 70)

    # ======================================================
    # Check required files
    # ======================================================

    if not os.path.exists(
        LLM_INPUT_FILE
    ):

        raise FileNotFoundError(
            "\nTrain-only LLM input file not found:\n"
            f"{LLM_INPUT_FILE}\n\n"
            "Run 07_build_train_only_llm_inputs.py first."
        )

    if not os.path.exists(
        CATALOGUE_FILE
    ):

        raise FileNotFoundError(
            "\nFashion LLM catalogue not found:\n"
            f"{CATALOGUE_FILE}\n\n"
            "Run 02_build_fashion_llm_catalogue.py first."
        )

    # ======================================================
    # Load catalogue
    # ======================================================

    print("\nLoading LLM catalogue:")
    print(CATALOGUE_FILE)

    with open(
        CATALOGUE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        catalogue = json.load(file)

    print(
        f"Catalogue products: {len(catalogue):,}"
    )

    # ------------------------------------------------------
    # Build product texts
    # ------------------------------------------------------

    catalogue_texts = []
    catalogue_asins = []

    for product in catalogue:

        asin = clean_text(
            product.get("asin")
        )

        if not asin:
            continue

        text = build_product_text(
            product
        )

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

    # ------------------------------------------------------
    # ASIN -> catalogue product lookup
    # ------------------------------------------------------

    catalogue_lookup = {
        clean_text(product["asin"]): product
        for product in catalogue
        if clean_text(
            product.get("asin")
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
    # Load TRAIN-ONLY user inputs
    # ======================================================

    print()
    print(
        "Loading train-only LLM user inputs:"
    )

    print(
        LLM_INPUT_FILE
    )

    with open(
        LLM_INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        llm_inputs = json.load(file)

    print(
        f"Users loaded: {len(llm_inputs):,}"
    )

    # ======================================================
    # Generate candidate sets
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
        # TRAIN history ASINs only
        # --------------------------------------------------

        history_asins = {
            clean_text(
                product.get("asin")
            )
            for product in history
            if clean_text(
                product.get("asin")
            )
        }

        # --------------------------------------------------
        # Build profile ONLY from TRAIN history
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

        user_vector = (
            vectorizer.transform(
                [user_profile]
            )
        )

        # --------------------------------------------------
        # Compute similarity against entire catalogue
        # --------------------------------------------------

        similarities = (
            cosine_similarity(
                user_vector,
                catalogue_matrix
            ).flatten()
        )

        # --------------------------------------------------
        # Rank products from most similar to least similar
        # --------------------------------------------------

        ranked_indices = (
            similarities.argsort()[::-1]
        )

        candidates = []

        candidate_number = 1

        for index in ranked_indices:

            asin = catalogue_asins[
                index
            ]

            # ------------------------------------------------
            # IMPORTANT:
            # Exclude ONLY products already visible in the
            # TRAIN history.
            #
            # Validation/test interactions were already
            # removed by file 07 and therefore never enter
            # this profile or exclusion set.
            # ------------------------------------------------

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
                    )
                }
            )

            candidate_number += 1

            if (
                len(candidates)
                >= TOP_K_CANDIDATES
            ):
                break

        # --------------------------------------------------
        # Candidate-set checks
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

        # Verify that no training item leaked into candidates
        candidate_asins = {
            candidate["asin"]
            for candidate in candidates
        }

        overlap = (
            candidate_asins
            & history_asins
        )

        if overlap:

            raise RuntimeError(
                "\nTRAIN-HISTORY CANDIDATE "
                "LEAKAGE DETECTED\n"
                f"User: {user_id}\n"
                f"Overlapping ASINs: "
                f"{sorted(overlap)}"
            )

        # --------------------------------------------------
        # Save user candidate set
        # --------------------------------------------------

        output_records.append(
            {
                "user_id": user_id,

                "history_length": len(
                    history
                ),

                "history_asins": sorted(
                    history_asins
                ),

                "candidate_count": (
                    candidate_count
                ),

                "candidates": candidates
            }
        )

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (
            user_number % 250 == 0
        ):

            print(
                f"Processed "
                f"{user_number:,} / "
                f"{len(llm_inputs):,} users"
            )

    # ======================================================
    # Final validation
    # ======================================================

    print()
    print("=" * 70)
    print(
        "VALIDATING CANDIDATE SETS"
    )
    print("=" * 70)

    if not output_records:

        raise RuntimeError(
            "No candidate records were created."
        )

    # Check duplicate users
    user_ids = [
        record["user_id"]
        for record in output_records
    ]

    if (
        len(user_ids)
        != len(set(user_ids))
    ):

        raise RuntimeError(
            "Duplicate user IDs detected "
            "in candidate output."
        )

    # Check duplicate candidates within each user
    duplicate_candidate_users = 0

    for record in output_records:

        candidate_asins = [
            candidate["asin"]
            for candidate
            in record["candidates"]
        ]

        if (
            len(candidate_asins)
            != len(set(candidate_asins))
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
        "No training-history ASINs appear "
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
    # Save output
    # ======================================================

    print()
    print("Saving candidate file:")
    print(OUTPUT_FILE)

    with open(
        OUTPUT_FILE,
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
    print("=" * 70)
    print(
        "TRAIN-ONLY FASHION CANDIDATES CREATED"
    )
    print("=" * 70)

    print(
        f"Users written:             "
        f"{len(output_records):,}"
    )

    print(
        f"Users skipped:             "
        f"{skipped_users:,}"
    )

    print(
        f"Target candidates/user:    "
        f"{TOP_K_CANDIDATES}"
    )

    print(
        f"Users with <50 candidates: "
        f"{users_with_fewer_than_k:,}"
    )

    if candidate_counts:

        print(
            f"Minimum candidate count:   "
            f"{min(candidate_counts):,}"
        )

        print(
            f"Maximum candidate count:   "
            f"{max(candidate_counts):,}"
        )

        print(
            f"Average candidate count:   "
            f"{sum(candidate_counts) / len(candidate_counts):.2f}"
        )

    print()
    print(
        f"Output file:\n{OUTPUT_FILE}"
    )

    # ======================================================
    # Example
    # ======================================================

    if output_records:

        print()
        print("=" * 70)
        print(
            "EXAMPLE TRAIN-ONLY CANDIDATE SET"
        )
        print("=" * 70)

        example = (
            output_records[0]
        )

        print(
            json.dumps(
                {
                    "user_id": (
                        example[
                            "user_id"
                        ]
                    ),
                    "history_length": (
                        example[
                            "history_length"
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
                    "first_five_candidates": (
                        example[
                            "candidates"
                        ][:5]
                    )
                },
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
        "09_generate_fashion_augmentations.py"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    create_train_only_candidates()