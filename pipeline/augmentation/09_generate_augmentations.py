"""
09_generate_augmentations.py

    python pipeline/augmentation/09_generate_augmentations.py --dataset fashion

    python pipeline/augmentation/09_generate_augmentations.py --dataset video_games

    python pipeline/augmentation/09_generate_augmentations.py --dataset movies_tv

IMPORTANT:
    Validation/test interactions are not supplied to Qwen.
"""

import json
import os
import re
import time
import argparse

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ==========================================================
# Dataset Configuration
# ==========================================================

DATASETS = {
    "fashion": {
        "folder": "Fashion",
        "prefix": "Fashion",
        "domain_label": "fashion products",
    },

    "video_games": {
        "folder": "Video Games",
        "prefix": "Video_Games",
        "domain_label": "video game products",
    },

    "movies_tv": {
        "folder": "Movies and TV Shows",
        "prefix": "Movies_TV",
        "domain_label": "movie and television products",
    },
}


# ==========================================================
# Model Configuration
# ==========================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

NUM_SELECTIONS = 3

MAX_NEW_TOKENS = 32

CHECKPOINT_EVERY = 50


# ==========================================================
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate LLM augmentations "
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


def compact_text(
    value,
    max_length=220
):

    value = clean_text(
        value
    )

    if (
        len(value)
        <= max_length
    ):

        return value

    return (
        value[
            :max_length
        ].rstrip()
        + "..."
    )


def format_history(history):

    lines = []

    for index, product in enumerate(
        history,
        start=1
    ):

        title = compact_text(
            product.get(
                "title",
                ""
            )
        )

        store = compact_text(
            product.get(
                "store",
                ""
            ),
            max_length=80
        )

        department = compact_text(
            product.get(
                "department",
                ""
            ),
            max_length=80
        )

        manufacturer = compact_text(
            product.get(
                "manufacturer",
                ""
            ),
            max_length=80
        )

        details = []

        if store:

            details.append(
                f"Brand/Store: {store}"
            )

        if department:

            details.append(
                f"Department: {department}"
            )

        if manufacturer:

            details.append(
                f"Manufacturer: {manufacturer}"
            )

        detail_text = ""

        if details:

            detail_text = (
                " | "
                + " | ".join(
                    details
                )
            )

        lines.append(
            f"{index}. "
            f"{title}"
            f"{detail_text}"
        )

    return "\n".join(
        lines
    )


def format_candidates(candidates):

    lines = []

    for candidate in candidates:

        candidate_id = clean_text(
            candidate.get(
                "candidate_id"
            )
        )

        title = compact_text(
            candidate.get(
                "title",
                ""
            )
        )

        store = compact_text(
            candidate.get(
                "store",
                ""
            ),
            max_length=80
        )

        department = compact_text(
            candidate.get(
                "department",
                ""
            ),
            max_length=80
        )

        details = []

        if store:

            details.append(
                f"Brand/Store: {store}"
            )

        if department:

            details.append(
                f"Department: {department}"
            )

        detail_text = ""

        if details:

            detail_text = (
                " | "
                + " | ".join(
                    details
                )
            )

        lines.append(
            f"{candidate_id}: "
            f"{title}"
            f"{detail_text}"
        )

    return "\n".join(
        lines
    )


# ==========================================================
# Prompt
# ==========================================================

def build_prompt(
    history,
    candidates,
    domain_label
):

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

    system_message = (
        "You are a recommendation assistant. "
        f"Your task is to infer a user's preferences for "
        f"{domain_label} from their historical interactions "
        "and select the most plausible next products from "
        "a fixed candidate set. "
        "You must only choose products from the supplied "
        "candidate list."
    )

    user_message = f"""
The following products are the user's chronological TRAINING history.

USER HISTORY
-------------
{history_text}

Below are 50 candidate products.

CANDIDATES
----------
{candidate_text}

Select exactly {NUM_SELECTIONS} candidate products that are the most plausible next items for this user.

Rules:
1. Choose only from the candidate IDs provided above.
2. Return exactly {NUM_SELECTIONS} different candidate IDs.
3. Do not return ASINs.
4. Do not explain your choices.
5. Do not include any extra words.
6. Return the IDs separated by spaces.

Required format:
C001 C014 C027
""".strip()

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


# ==========================================================
# Candidate Parsing
# ==========================================================

def parse_candidate_ids(
    raw_response,
    valid_candidate_ids
):

    matches = re.findall(
        r"\bC\d{3}\b",
        raw_response.upper()
    )

    valid_ids = []
    seen = set()

    for candidate_id in matches:

        if (
            candidate_id
            not in valid_candidate_ids
        ):
            continue

        if (
            candidate_id
            in seen
        ):
            continue

        seen.add(
            candidate_id
        )

        valid_ids.append(
            candidate_id
        )

    return valid_ids[
        :NUM_SELECTIONS
    ]


# ==========================================================
# Load / Save
# ==========================================================

def load_json(path):

    print(
        "Loading:"
    )

    print(
        path
    )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


def save_results(
    results,
    output_file
):

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )


# ==========================================================
# Main
# ==========================================================

def generate_augmentations():

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

    domain_label = (
        config[
            "domain_label"
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

    export_dir = os.path.join(
        project_dir,
        "datasets",
        dataset_folder,
        "exports"
    )

    llm_input_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_train_inputs.json"
    )

    candidate_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_train_candidates.json"
    )

    output_file = os.path.join(
        export_dir,
        f"{dataset_prefix}_llm_augmentations_raw.json"
    )

    # ======================================================
    # Header
    # ======================================================

    print(
        "=" * 70
    )

    print(
        f"GENERATING "
        f"{dataset_folder.upper()} "
        f"LLM AUGMENTATIONS"
    )

    print(
        "=" * 70
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
    # Load Inputs
    # ======================================================

    print()
    print(
        "Loading train-only LLM inputs:"
    )

    llm_inputs = load_json(
        llm_input_file
    )

    print(
        f"Users loaded: "
        f"{len(llm_inputs):,}"
    )

    print()
    print(
        "Loading train-only candidate sets:"
    )

    candidate_records = load_json(
        candidate_file
    )

    print(
        f"Candidate records loaded: "
        f"{len(candidate_records):,}"
    )

    # ------------------------------------------------------
    # User lookups
    # ------------------------------------------------------

    input_lookup = {
        record[
            "user_id"
        ]: record

        for record
        in llm_inputs

        if record.get(
            "user_id"
        )
    }

    candidate_lookup = {
        record[
            "user_id"
        ]: record

        for record
        in candidate_records

        if record.get(
            "user_id"
        )
    }

    common_users = sorted(
        set(
            input_lookup
        )
        & set(
            candidate_lookup
        )
    )

    print()
    print(
        f"Users available in both files: "
        f"{len(common_users):,}"
    )

    if not common_users:

        raise RuntimeError(
            "No common users found between "
            "input and candidate files."
        )

    # ======================================================
    # Load Qwen
    # ======================================================

    print()
    print(
        "Loading tokenizer..."
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
            dtype=torch.float16,
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
    # Resume Existing Output if Present
    # ======================================================

    results = []

    completed_users = set()

    if os.path.exists(
        output_file
    ):

        print()
        print(
            "Existing output file found."
        )

        print(
            "Loading previous progress..."
        )

        with open(
            output_file,
            "r",
            encoding="utf-8"
        ) as file:

            results = json.load(
                file
            )

        completed_users = {
            record[
                "user_id"
            ]

            for record
            in results

            if record.get(
                "user_id"
            )
        }

        print(
            f"Previously completed users: "
            f"{len(completed_users):,}"
        )

    # ======================================================
    # Generation Statistics
    # ======================================================

    successful_users = sum(
        1
        for record
        in results
        if record.get(
            "status"
        ) == "success"
    )

    invalid_users = sum(
        1
        for record
        in results
        if record.get(
            "status"
        ) != "success"
    )

    newly_processed = 0

    start_time = (
        time.time()
    )

    # ======================================================
    # Generate Per User
    # ======================================================

    for user_number, user_id in enumerate(
        common_users,
        start=1
    ):

        if (
            user_id
            in completed_users
        ):
            continue

        input_record = (
            input_lookup[
                user_id
            ]
        )

        candidate_record = (
            candidate_lookup[
                user_id
            ]
        )

        history = input_record.get(
            "history",
            []
        )

        candidates = candidate_record.get(
            "candidates",
            []
        )

        if not history or not candidates:

            results.append(
                {
                    "user_id": (
                        user_id
                    ),

                    "status": (
                        "skipped"
                    ),

                    "reason": (
                        "Missing history or candidates"
                    ),

                    "selected_candidates": [],
                }
            )

            invalid_users += 1
            newly_processed += 1

            continue

        # --------------------------------------------------
        # Candidate lookup
        # --------------------------------------------------

        candidate_id_lookup = {
            candidate[
                "candidate_id"
            ]: candidate

            for candidate
            in candidates

            if candidate.get(
                "candidate_id"
            )
        }

        valid_candidate_ids = set(
            candidate_id_lookup.keys()
        )

        # --------------------------------------------------
        # Build prompt
        # --------------------------------------------------

        messages = build_prompt(
            history,
            candidates,
            domain_label
        )

        prompt_text = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        model_inputs = tokenizer(
            prompt_text,
            return_tensors="pt"
        )

        model_inputs = {
            key: value.to(
                model.device
            )

            for key, value
            in model_inputs.items()
        }

        # --------------------------------------------------
        # Generate
        # --------------------------------------------------

        with torch.inference_mode():

            generated_ids = (
                model.generate(
                    **model_inputs,
                    max_new_tokens=(
                        MAX_NEW_TOKENS
                    ),
                    do_sample=False,
                    pad_token_id=(
                        tokenizer.eos_token_id
                    ),
                )
            )

        input_length = (
            model_inputs[
                "input_ids"
            ].shape[1]
        )

        new_tokens = (
            generated_ids[
                0,
                input_length:
            ]
        )

        raw_response = (
            tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            ).strip()
        )

        # --------------------------------------------------
        # Validate candidate IDs
        # --------------------------------------------------

        selected_ids = (
            parse_candidate_ids(
                raw_response,
                valid_candidate_ids
            )
        )

        selected_candidates = []

        for candidate_id in selected_ids:

            candidate = (
                candidate_id_lookup[
                    candidate_id
                ]
            )

            selected_candidates.append(
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
                        candidate.get(
                            "title",
                            ""
                        )
                    ),

                    "similarity_score": (
                        candidate.get(
                            "similarity_score"
                        )
                    ),
                }
            )

        # --------------------------------------------------
        # Determine status
        # --------------------------------------------------

        if (
            len(
                selected_candidates
            )
            == NUM_SELECTIONS
        ):

            status = (
                "success"
            )

            successful_users += 1

        else:

            status = (
                "invalid_output"
            )

            invalid_users += 1

        # --------------------------------------------------
        # Save record
        # --------------------------------------------------

        results.append(
            {
                "user_id": (
                    user_id
                ),

                "history_length": len(
                    history
                ),

                "candidate_count": len(
                    candidates
                ),

                "raw_model_response": (
                    raw_response
                ),

                "validated_candidate_ids": (
                    selected_ids
                ),

                "selected_candidates": (
                    selected_candidates
                ),

                "status": (
                    status
                ),
            }
        )

        newly_processed += 1

        # ==================================================
        # Progress
        # ==================================================

        if (
            newly_processed
            % 10
            == 0
        ):

            print(
                f"Processed "
                f"{newly_processed:,} new users "
                f"| overall position "
                f"{user_number:,}/"
                f"{len(common_users):,}"
                f" | successes "
                f"{successful_users:,}"
            )

        # ==================================================
        # Periodic Checkpoint
        # ==================================================

        if (
            newly_processed
            % CHECKPOINT_EVERY
            == 0
        ):

            save_results(
                results,
                output_file
            )

            print(
                f"Checkpoint saved "
                f"({len(results):,} total records)"
            )

    # ======================================================
    # Final Save
    # ======================================================

    save_results(
        results,
        output_file
    )

    elapsed = (
        time.time()
        - start_time
    )

    # ======================================================
    # Final Summary
    # ======================================================

    print()
    print(
        "=" * 70
    )

    print(
        f"{dataset_folder.upper()} "
        f"LLM AUGMENTATION GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Dataset:                   "
        f"{dataset_folder}"
    )

    print(
        f"Total eligible users:      "
        f"{len(common_users):,}"
    )

    print(
        f"Total output records:      "
        f"{len(results):,}"
    )

    print(
        f"Successful generations:    "
        f"{successful_users:,}"
    )

    print(
        f"Invalid/skipped outputs:   "
        f"{invalid_users:,}"
    )

    if newly_processed:

        print(
            f"Users processed this run:  "
            f"{newly_processed:,}"
        )

        print(
            f"Average seconds/user:      "
            f"{elapsed / newly_processed:.2f}"
        )

    print()
    print(
        f"Output file:\n"
        f"{output_file}"
    )

    # ======================================================
    # Example
    # ======================================================

    successful_examples = [
        record
        for record
        in results
        if record.get(
            "status"
        ) == "success"
    ]

    if successful_examples:

        print()
        print(
            "=" * 70
        )

        print(
            "EXAMPLE GENERATED AUGMENTATION"
        )

        print(
            "=" * 70
        )

        print(
            json.dumps(
                successful_examples[
                    0
                ],
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

    print()
    print(
        "Next step:"
    )

    print(
        "10_validate_augmentations.py "
        f"--dataset {dataset_key}"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    generate_augmentations()