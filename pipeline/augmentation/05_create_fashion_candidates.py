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
    "Fashion_llm_inputs.json"
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
    "Fashion_llm_candidates.json"
)


# ==========================================================
# Configuration
# ==========================================================

TOP_K_CANDIDATES = 50


# ==========================================================
# Helper Functions
# ==========================================================

def clean_text(value):

    if value is None:
        return ""

    return str(value).strip()


def join_list(values):

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

    print("=" * 70)
    print("CREATING FASHION LLM CANDIDATES")
    print("=" * 70)

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

    # ------------------------------------------------------
    # ASIN -> catalogue product lookup
    # ------------------------------------------------------

    catalogue_lookup = {
        product["asin"]: product
        for product in catalogue
        if product.get("asin")
    }

    # ======================================================
    # Build TF-IDF catalogue matrix
    # ======================================================

    print(
        "\nBuilding TF-IDF catalogue representation..."
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000
    )

    catalogue_matrix = vectorizer.fit_transform(
        catalogue_texts
    )

    print(
        "TF-IDF matrix shape:",
        catalogue_matrix.shape
    )

    # ======================================================
    # Load LLM user inputs
    # ======================================================

    print("\nLoading LLM user inputs:")
    print(LLM_INPUT_FILE)

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
    # Generate candidates
    # ======================================================

    output_records = []

    skipped_users = 0

    for user_number, user_record in enumerate(
        llm_inputs,
        start=1
    ):

        user_id = user_record.get(
            "user_id"
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
            product.get("asin")
            for product in history
            if product.get("asin")
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
            [user_profile]
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

        ranked_indices = similarities.argsort()[::-1]

        candidates = []

        candidate_number = 1

        for index in ranked_indices:

            asin = catalogue_asins[index]

            # Never recommend something already
            # present in the real user history
            if asin in history_asins:
                continue

            product = catalogue_lookup[
                asin
            ]

            candidate_id = (
                f"C{candidate_number:03d}"
            )

            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "asin": asin,
                    "similarity_score": float(
                        similarities[index]
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
                    ),
                    "features": product.get(
                        "features",
                        []
                    )
                }
            )

            candidate_number += 1

            if len(candidates) >= TOP_K_CANDIDATES:
                break

        # --------------------------------------------------
        # Save user candidate set
        # --------------------------------------------------

        output_records.append(
            {
                "user_id": user_id,
                "history_length": len(
                    history
                ),
                "history_asins": list(
                    history_asins
                ),
                "candidate_count": len(
                    candidates
                ),
                "candidates": candidates
            }
        )

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if user_number % 250 == 0:

            print(
                f"Processed "
                f"{user_number:,} / "
                f"{len(llm_inputs):,} users"
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
            output_records,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ======================================================
    # Summary
    # ======================================================

    print("\n" + "=" * 70)
    print("FASHION CANDIDATES CREATED")
    print("=" * 70)

    print(
        f"Users written:       {len(output_records):,}"
    )

    print(
        f"Users skipped:       {skipped_users:,}"
    )

    print(
        f"Candidates per user: {TOP_K_CANDIDATES}"
    )

    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )

    # ======================================================
    # Example
    # ======================================================

    if output_records:

        print("\nExample candidate set:")
        print("-" * 70)

        example = output_records[0]

        print(
            json.dumps(
                {
                    "user_id": example["user_id"],
                    "history_asins": example[
                        "history_asins"
                    ],
                    "candidates": example[
                        "candidates"
                    ][:5]
                },
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

    create_candidates()