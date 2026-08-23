"""
06_test_base_llm.py

Tests Qwen2.5-3B-Instruct on a small number of real users
for one of the supported Amazon datasets.

Supported datasets:
    fashion
    video_games
    movies_tv

Examples:

    python pipeline/augmentation/06_test_base_llm.py --dataset fashion

    python pipeline/augmentation/06_test_base_llm.py --dataset video_games

    python pipeline/augmentation/06_test_base_llm.py --dataset movies_tv
"""

import json
import os
import re
import argparse
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# ==========================================================
# Configuration
# ==========================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

NUM_USERS_TO_TEST = 5
NUM_RECOMMENDATIONS = 3


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
            "Test the base Qwen model on "
            "real users from an Amazon dataset."
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

    parser.add_argument(
        "--num-users",
        type=int,
        default=NUM_USERS_TO_TEST,
        help=(
            "Number of users to test. "
            f"Default: {NUM_USERS_TO_TEST}"
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


def format_history(history):
    """
    Format chronological user history for the prompt.
    """

    lines = []

    for index, product in enumerate(
        history,
        start=1
    ):

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

        line = (
            f"{index}. {title}"
        )

        if store:

            line += (
                f" | Store: {store}"
            )

        if department:

            line += (
                f" | Department: "
                f"{department}"
            )

        if manufacturer:

            line += (
                f" | Manufacturer: "
                f"{manufacturer}"
            )

        lines.append(
            line
        )

    return "\n".join(
        lines
    )


def format_candidates(candidates):
    """
    Format the user's candidate pool.
    """

    lines = []

    for candidate in candidates:

        candidate_id = clean_text(
            candidate.get(
                "candidate_id"
            )
        )

        title = clean_text(
            candidate.get(
                "title"
            )
        )

        store = clean_text(
            candidate.get(
                "store"
            )
        )

        department = clean_text(
            candidate.get(
                "department"
            )
        )

        manufacturer = clean_text(
            candidate.get(
                "manufacturer"
            )
        )

        line = (
            f"{candidate_id} - {title}"
        )

        if store:

            line += (
                f" | Store: {store}"
            )

        if department:

            line += (
                f" | Department: "
                f"{department}"
            )

        if manufacturer:

            line += (
                f" | Manufacturer: "
                f"{manufacturer}"
            )

        lines.append(
            line
        )

    return "\n".join(
        lines
    )


def extract_candidate_ids(
    response,
    valid_candidate_ids
):
    """
    Extract candidate IDs such as C001, C014, C027
    and reject IDs that were not in the candidate set.
    """

    found_ids = re.findall(
        r"\bC\d{3}\b",
        response.upper()
    )

    selected = []
    seen = set()

    for candidate_id in found_ids:

        if (
            candidate_id
            in valid_candidate_ids
            and candidate_id
            not in seen
        ):

            selected.append(
                candidate_id
            )

            seen.add(
                candidate_id
            )

    return selected


# ==========================================================
# Main
# ==========================================================

def test_base_llm():

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

    num_users_to_test = (
        args.num_users
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

    export_dir = os.path.join(
        project_dir,
        "datasets",
        dataset_folder,
        "exports"
    )

    llm_input_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_inputs.json"
    )

    candidate_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_candidates.json"
    )

    # ======================================================
    # Validate Input Files
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
        candidate_file
    ):

        raise FileNotFoundError(
            "\nCandidate file not found:\n"
            f"{candidate_file}\n\n"
            "Run 05_create_candidates.py "
            f"--dataset {dataset_key} first."
        )

    # ======================================================
    # Header
    # ======================================================

    print(
        "=" * 70
    )

    print(
        f"TESTING BASE QWEN MODEL ON "
        f"REAL {dataset_folder.upper()} USERS"
    )

    print(
        "=" * 70
    )

    # ======================================================
    # Load user histories
    # ======================================================

    print(
        "\nLoading LLM inputs:"
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
    # Load candidate sets
    # ======================================================

    print(
        "\nLoading candidate sets:"
    )

    print(
        candidate_file
    )

    with open(
        candidate_file,
        "r",
        encoding="utf-8"
    ) as file:

        candidate_records = (
            json.load(
                file
            )
        )

    print(
        f"Candidate records loaded: "
        f"{len(candidate_records):,}"
    )

    # ------------------------------------------------------
    # Build lookups
    # ------------------------------------------------------

    input_lookup = {
        record["user_id"]: record
        for record in llm_inputs
        if record.get(
            "user_id"
        )
    }

    candidate_lookup = {
        record["user_id"]: record
        for record
        in candidate_records
        if record.get(
            "user_id"
        )
    }

    common_users = [
        user_id
        for user_id in input_lookup
        if user_id
        in candidate_lookup
    ]

    print(
        f"Users available in both files: "
        f"{len(common_users):,}"
    )

    if not common_users:

        raise RuntimeError(
            "No users exist in both "
            "the LLM input and candidate files."
        )

    # ======================================================
    # GPU Check
    # ======================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "\nCUDA GPU not available.\n"
            "Run this script in Colab with a GPU runtime "
            "or submit it as a GPU job on the HPC cluster."
        )

    print()
    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )

    # ======================================================
    # Load Qwen
    # ======================================================

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    print(
        "Loading model..."
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="auto"
        )
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    print(
        "Model device:",
        next(
            model.parameters()
        ).device
    )

    # ======================================================
    # Test small set of users
    # ======================================================

    test_users = (
        common_users[
            :num_users_to_test
        ]
    )

    for (
        user_number,
        user_id
    ) in enumerate(
        test_users,
        start=1
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"USER {user_number}: "
            f"{user_id}"
        )

        print(
            "=" * 70
        )

        history = (
            input_lookup[
                user_id
            ][
                "history"
            ]
        )

        candidates = (
            candidate_lookup[
                user_id
            ][
                "candidates"
            ]
        )

        # --------------------------------------------------
        # Format prompt content
        # --------------------------------------------------

        history_text = (
            format_history(
                history
            )
        )

        candidate_text = (
            format_candidates(
                candidates
            )
        )

        candidate_by_id = {
            candidate[
                "candidate_id"
            ]: candidate
            for candidate
            in candidates
        }

        valid_candidate_ids = set(
            candidate_by_id.keys()
        )

        # ==================================================
        # Prompt
        # ==================================================

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a product recommendation assistant. "
                    "You must choose products only from the "
                    "candidate list provided. "
                    "Never invent a product or candidate ID."
                ),
            },
            {
                "role": "user",
                "content": f"""
The following is a user's chronological product interaction history.
The items are ordered from oldest to newest.

Give greater importance to the user's more recent interactions when
determining their current preferences.

USER HISTORY
-------------
{history_text}

CANDIDATE PRODUCTS
------------------
{candidate_text}

Select exactly {NUM_RECOMMENDATIONS} candidate products that would
be the most plausible additions to this user's product history.

Rules:
1. Choose only candidate IDs from the list above.
2. Do not invent products or IDs.
3. Do not repeat an ID.
4. Return exactly {NUM_RECOMMENDATIONS} candidate IDs.
5. Return only the IDs separated by spaces.

Example:
C001 C014 C027
""",
            },
        ]

        # ==================================================
        # Apply chat template
        # ==================================================

        prompt_text = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt"
        ).to(
            model.device
        )

        # ==================================================
        # Generate
        # ==================================================

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=(
                    tokenizer.eos_token_id
                ),
            )

        generated_tokens = outputs[
            0,
            inputs[
                "input_ids"
            ].shape[1]:
        ]

        response = (
            tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()
        )

        # ==================================================
        # Validate output
        # ==================================================

        selected_ids = (
            extract_candidate_ids(
                response,
                valid_candidate_ids
            )
        )

        print(
            "\nRaw model response:"
        )

        print(
            response
        )

        print(
            "\nValidated candidate IDs:"
        )

        print(
            selected_ids
        )

        if (
            len(selected_ids)
            != NUM_RECOMMENDATIONS
        ):

            print(
                "\nWARNING: Expected "
                f"{NUM_RECOMMENDATIONS} valid IDs, "
                f"but received {len(selected_ids)}."
            )

        # ==================================================
        # Map candidate IDs back to ASINs
        # ==================================================

        selected_products = []

        for candidate_id in (
            selected_ids
        ):

            candidate = (
                candidate_by_id[
                    candidate_id
                ]
            )

            selected_products.append(
                {
                    "candidate_id": (
                        candidate_id
                    ),
                    "asin": (
                        candidate[
                            "asin"
                        ]
                    ),
                    "title": (
                        candidate[
                            "title"
                        ]
                    ),
                }
            )

        print(
            "\nMapped products:"
        )

        print(
            json.dumps(
                selected_products,
                indent=2,
                ensure_ascii=False
            )
        )

    print(
        "\n" + "=" * 70
    )

    print(
        f"BASE LLM "
        f"{dataset_folder.upper()} "
        f"REAL-USER TEST COMPLETE"
    )

    print(
        "=" * 70
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    test_base_llm()