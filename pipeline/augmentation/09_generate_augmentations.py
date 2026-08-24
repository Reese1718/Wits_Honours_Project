"""
09_generate_augmentations.py

Runs Qwen2.5-3B-Instruct over the full eligible user set
for one of the supported datasets.

Supported datasets:
    fashion
    video_games
    movies_tv

This version performs BATCHED inference for substantially
better GPU utilisation.

Inputs:
    {prefix}_llm_train_inputs.json
    {prefix}_llm_train_candidates.json

For each user:
    1. Give Qwen the TRAIN-ONLY user history.
    2. Give Qwen the candidate products.
    3. Ask Qwen to select exactly 3 candidate IDs.
    4. Validate the returned candidate IDs.
    5. Map valid candidate IDs back to real ASINs.
    6. Save the generated augmentation records.

Output:
    {prefix}_llm_augmentations_raw.json

Examples:

    python pipeline/augmentation/09_generate_augmentations.py \
        --dataset fashion

    python pipeline/augmentation/09_generate_augmentations.py \
        --dataset video_games

    python pipeline/augmentation/09_generate_augmentations.py \
        --dataset movies_tv

Optional:

    python pipeline/augmentation/09_generate_augmentations.py \
        --dataset video_games \
        --batch-size 16

IMPORTANT:
    Validation/test interactions are NEVER supplied to Qwen.

    Existing output files are automatically loaded so interrupted
    runs can resume without regenerating completed users.
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
# Model / Generation Configuration
# ==========================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

NUM_SELECTIONS = 3

MAX_NEW_TOKENS = 32

# A100 40 GB:
# 16 is a safe starting point.
DEFAULT_BATCH_SIZE = 16

# Saving the entire growing JSON file to Google Drive every
# 50 users causes unnecessary I/O overhead.
DEFAULT_CHECKPOINT_EVERY = 500


# ==========================================================
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate batched LLM augmentations "
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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of users processed simultaneously. "
            f"Default: {DEFAULT_BATCH_SIZE}"
        ),
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help=(
            "Save progress after approximately this many "
            "newly processed users. "
            f"Default: {DEFAULT_CHECKPOINT_EVERY}"
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

    if len(value) <= max_length:
        return value

    return (
        value[:max_length].rstrip()
        + "..."
    )


def format_history(history):
    """
    Format one user's TRAIN-ONLY history.
    """

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
                + " | ".join(details)
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
    """
    Format one user's candidate set.
    """

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
                + " | ".join(details)
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

    history_text = format_history(
        history
    )

    candidate_text = format_candidates(
        candidates
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

Below are candidate products.

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
    """
    Extract valid unique candidate IDs while preserving
    the order returned by Qwen.
    """

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

        if candidate_id in seen:
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
            f"\nRequired file not found:\n"
            f"{path}"
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

    # Write to a temporary file first.
    # This reduces the risk of corrupting the real checkpoint
    # if Colab disconnects during a write.

    temporary_file = (
        output_file
        + ".tmp"
    )

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    os.replace(
        temporary_file,
        output_file
    )


# ==========================================================
# Main
# ==========================================================

def generate_augmentations():

    args = parse_args()

    dataset_key = (
        args.dataset
    )

    batch_size = (
        args.batch_size
    )

    checkpoint_every = (
        args.checkpoint_every
    )

    if batch_size < 1:

        raise ValueError(
            "--batch-size must be at least 1."
        )

    if checkpoint_every < 1:

        raise ValueError(
            "--checkpoint-every must be at least 1."
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

    print(
        f"\nBatch size:        "
        f"{batch_size}"
    )

    print(
        f"Checkpoint every:  "
        f"{checkpoint_every:,} users"
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

    gpu_name = (
        torch.cuda.get_device_name(
            0
        )
    )

    print()

    print(
        "GPU:",
        gpu_name
    )

    # A100 supports BF16 extremely well.

    use_bfloat16 = (
        torch.cuda.is_bf16_supported()
    )

    model_dtype = (
        torch.bfloat16
        if use_bfloat16
        else torch.float16
    )

    print(
        "Model dtype:",
        model_dtype
    )

    # Allow TensorFloat-32 operations where applicable.
    torch.backends.cuda.matmul.allow_tf32 = True

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
        clean_text(
            record.get(
                "user_id"
            )
        ): record

        for record in llm_inputs

        if clean_text(
            record.get(
                "user_id"
            )
        )
    }

    candidate_lookup = {
        clean_text(
            record.get(
                "user_id"
            )
        ): record

        for record in candidate_records

        if clean_text(
            record.get(
                "user_id"
            )
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

    # Decoder-only LLMs should use left padding during
    # batched generation.

    tokenizer.padding_side = (
        "left"
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        "Loading model..."
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=model_dtype,
            device_map="auto"
        )
    )

    model.eval()

    model_device = next(
        model.parameters()
    ).device

    print(
        "Model loaded successfully."
    )

    print(
        "Model device:",
        model_device
    )

    # ======================================================
    # Resume Existing Output
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
            clean_text(
                record.get(
                    "user_id"
                )
            )

            for record in results

            if clean_text(
                record.get(
                    "user_id"
                )
            )
        }

        print(
            f"Previously completed users: "
            f"{len(completed_users):,}"
        )

    # ======================================================
    # Determine Remaining Users
    # ======================================================

    remaining_users = [
        user_id
        for user_id in common_users
        if user_id not in completed_users
    ]

    print()

    print(
        f"Users remaining: "
        f"{len(remaining_users):,}"
    )

    if not remaining_users:

        print(
            "\nAll eligible users are already complete."
        )

        return

    # ======================================================
    # Existing Statistics
    # ======================================================

    successful_users = sum(
        1
        for record in results
        if record.get(
            "status"
        ) == "success"
    )

    invalid_users = sum(
        1
        for record in results
        if record.get(
            "status"
        ) != "success"
    )

    newly_processed = 0

    users_since_checkpoint = 0

    start_time = (
        time.time()
    )

    total_remaining = len(
        remaining_users
    )

    # ======================================================
    # Batched Generation
    # ======================================================

    for batch_start in range(
        0,
        total_remaining,
        batch_size
    ):

        batch_user_ids = (
            remaining_users[
                batch_start:
                batch_start
                + batch_size
            ]
        )

        # --------------------------------------------------
        # Build batch metadata
        # --------------------------------------------------

        batch_prompt_texts = []

        batch_metadata = []

        for user_id in batch_user_ids:

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

            candidates = (
                candidate_record.get(
                    "candidates",
                    []
                )
            )

            # ----------------------------------------------
            # Handle invalid / empty user record
            # ----------------------------------------------

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
                            "Missing history "
                            "or candidates"
                        ),

                        "selected_candidates": [],
                    }
                )

                invalid_users += 1
                newly_processed += 1
                users_since_checkpoint += 1

                continue

            candidate_id_lookup = {
                clean_text(
                    candidate.get(
                        "candidate_id"
                    )
                ): candidate

                for candidate in candidates

                if clean_text(
                    candidate.get(
                        "candidate_id"
                    )
                )
            }

            valid_candidate_ids = set(
                candidate_id_lookup.keys()
            )

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

            batch_prompt_texts.append(
                prompt_text
            )

            batch_metadata.append(
                {
                    "user_id": (
                        user_id
                    ),

                    "history": (
                        history
                    ),

                    "candidates": (
                        candidates
                    ),

                    "candidate_id_lookup": (
                        candidate_id_lookup
                    ),

                    "valid_candidate_ids": (
                        valid_candidate_ids
                    ),
                }
            )

        # --------------------------------------------------
        # Some records may have been skipped.
        # --------------------------------------------------

        if batch_prompt_texts:

            # ==================================================
            # Tokenize Entire Batch
            # ==================================================

            model_inputs = tokenizer(
                batch_prompt_texts,
                return_tensors="pt",
                padding=True,
            )

            model_inputs = {
                key: value.to(
                    model_device
                )

                for key, value
                in model_inputs.items()
            }

            padded_input_length = (
                model_inputs[
                    "input_ids"
                ].shape[1]
            )

            # ==================================================
            # Generate Entire Batch
            # ==================================================

            with torch.inference_mode():

                generated_ids = (
                    model.generate(
                        **model_inputs,
                        max_new_tokens=(
                            MAX_NEW_TOKENS
                        ),
                        do_sample=False,
                        pad_token_id=(
                            tokenizer.pad_token_id
                        ),
                        eos_token_id=(
                            tokenizer.eos_token_id
                        ),
                    )
                )

            # ==================================================
            # Decode Each User
            # ==================================================

            for output_index, metadata in enumerate(
                batch_metadata
            ):

                new_tokens = (
                    generated_ids[
                        output_index,
                        padded_input_length:
                    ]
                )

                raw_response = (
                    tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=True
                    ).strip()
                )

                valid_candidate_ids = (
                    metadata[
                        "valid_candidate_ids"
                    ]
                )

                candidate_id_lookup = (
                    metadata[
                        "candidate_id_lookup"
                    ]
                )

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

                # ------------------------------------------
                # Status
                # ------------------------------------------

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

                # ------------------------------------------
                # Save result
                # ------------------------------------------

                results.append(
                    {
                        "user_id": (
                            metadata[
                                "user_id"
                            ]
                        ),

                        "history_length": len(
                            metadata[
                                "history"
                            ]
                        ),

                        "candidate_count": len(
                            metadata[
                                "candidates"
                            ]
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
                users_since_checkpoint += 1

            # Free batch tensors before the next iteration.
            del model_inputs
            del generated_ids

        # ==================================================
        # Progress
        # ==================================================

        elapsed = (
            time.time()
            - start_time
        )

        if newly_processed:

            average_seconds = (
                elapsed
                / newly_processed
            )

            remaining_count = (
                total_remaining
                - newly_processed
            )

            estimated_remaining_seconds = (
                remaining_count
                * average_seconds
            )

            print(
                f"Processed "
                f"{newly_processed:,}/"
                f"{total_remaining:,} new users "
                f"| total saved "
                f"{len(results):,} "
                f"| successes "
                f"{successful_users:,} "
                f"| avg "
                f"{average_seconds:.3f}s/user "
                f"| estimated remaining "
                f"{estimated_remaining_seconds / 3600:.2f}h"
            )

        # ==================================================
        # Periodic Checkpoint
        # ==================================================

        if (
            users_since_checkpoint
            >= checkpoint_every
        ):

            save_results(
                results,
                output_file
            )

            print(
                f"Checkpoint saved "
                f"({len(results):,} total records)"
            )

            users_since_checkpoint = 0

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
        f"Batch size:                "
        f"{batch_size}"
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
            f"{elapsed / newly_processed:.3f}"
        )

        print(
            f"Total generation time:     "
            f"{elapsed / 3600:.2f} hours"
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
        for record in results
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
                successful_examples[0],
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