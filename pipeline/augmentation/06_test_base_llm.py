import json
import os
import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================================================
# Configuration
# ==========================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

NUM_USERS_TO_TEST = 5
NUM_RECOMMENDATIONS = 3


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

CANDIDATE_FILE = os.path.join(
    PROJECT_DIR,
    "datasets",
    "Fashion",
    "exports",
    "Fashion_llm_candidates.json"
)


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

        line = f"{index}. {title}"

        if store:
            line += f" | Store: {store}"

        if department:
            line += f" | Department: {department}"

        if manufacturer:
            line += f" | Manufacturer: {manufacturer}"

        lines.append(line)

    return "\n".join(lines)


def format_candidates(candidates):
    """
    Format the user's candidate pool.
    """

    lines = []

    for candidate in candidates:

        candidate_id = clean_text(
            candidate.get("candidate_id")
        )

        title = clean_text(
            candidate.get("title")
        )

        store = clean_text(
            candidate.get("store")
        )

        department = clean_text(
            candidate.get("department")
        )

        manufacturer = clean_text(
            candidate.get("manufacturer")
        )

        line = f"{candidate_id} - {title}"

        if store:
            line += f" | Store: {store}"

        if department:
            line += f" | Department: {department}"

        if manufacturer:
            line += f" | Manufacturer: {manufacturer}"

        lines.append(line)

    return "\n".join(lines)


def extract_candidate_ids(
    response,
    valid_candidate_ids
):
    """
    Extract candidate IDs such as C001, C014, C027
    and reject IDs that were not in the candidate set.
    """

    found_ids = re.findall(
        r"C\d{3}",
        response.upper()
    )

    selected = []
    seen = set()

    for candidate_id in found_ids:

        if (
            candidate_id in valid_candidate_ids
            and candidate_id not in seen
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

    print("=" * 70)
    print("TESTING BASE QWEN MODEL ON REAL FASHION USERS")
    print("=" * 70)

    # ======================================================
    # Load user histories
    # ======================================================

    print("\nLoading LLM inputs:")
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
    # Load candidate sets
    # ======================================================

    print("\nLoading candidate sets:")
    print(CANDIDATE_FILE)

    with open(
        CANDIDATE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        candidate_records = json.load(file)

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
    }

    candidate_lookup = {
        record["user_id"]: record
        for record in candidate_records
    }

    common_users = [
        user_id
        for user_id in input_lookup
        if user_id in candidate_lookup
    ]

    print(
        f"Users available in both files: "
        f"{len(common_users):,}"
    )

    # ======================================================
    # Load Qwen
    # ======================================================

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    print(
        "Model device:",
        model.device
    )

    # ======================================================
    # Test small set of users
    # ======================================================

    test_users = common_users[
        :NUM_USERS_TO_TEST
    ]

    for user_number, user_id in enumerate(
        test_users,
        start=1
    ):

        print("\n" + "=" * 70)

        print(
            f"USER {user_number}: {user_id}"
        )

        print("=" * 70)

        history = input_lookup[
            user_id
        ]["history"]

        candidates = candidate_lookup[
            user_id
        ]["candidates"]

        # --------------------------------------------------
        # Format prompt content
        # --------------------------------------------------

        history_text = format_history(
            history
        )

        candidate_text = format_candidates(
            candidates
        )

        candidate_by_id = {
            candidate["candidate_id"]: candidate
            for candidate in candidates
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
                )
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
"""
            }
        ]

        # ==================================================
        # Apply chat template
        # ==================================================

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
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

        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False
            )

        generated_tokens = outputs[
            0,
            inputs["input_ids"].shape[1]:
        ]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        # ==================================================
        # Validate output
        # ==================================================

        selected_ids = extract_candidate_ids(
            response,
            valid_candidate_ids
        )

        print("\nRaw model response:")
        print(response)

        print("\nValidated candidate IDs:")
        print(selected_ids)

        # ==================================================
        # Map candidate IDs back to ASINs
        # ==================================================

        selected_products = []

        for candidate_id in selected_ids:

            candidate = candidate_by_id[
                candidate_id
            ]

            selected_products.append(
                {
                    "candidate_id": candidate_id,
                    "asin": candidate["asin"],
                    "title": candidate["title"]
                }
            )

        print("\nMapped products:")

        print(
            json.dumps(
                selected_products,
                indent=2,
                ensure_ascii=False
            )
        )

    print("\n" + "=" * 70)
    print("BASE LLM REAL-USER TEST COMPLETE")
    print("=" * 70)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    test_base_llm()