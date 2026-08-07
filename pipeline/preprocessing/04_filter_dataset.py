import json
import os
from collections import Counter, defaultdict

# ==========================================================
# Project Configuration
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASETS_ROOT = os.path.join(BASE_DIR, "datasets")

CLEAN_DATASETS = {
    "Fashion": "Fashion_clean.jsonl",
    "Movies and TV Shows": "Movies_and_TV_Shows_clean.jsonl",
    "Video Games": "Video_Games_clean.jsonl"
}

MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset(file_path):

    interactions = []

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                interactions.append(json.loads(line))

    return interactions


# ==========================================================
# Single-Pass Filtering
# ==========================================================

def filter_dataset(interactions,
                   min_user=5,
                   min_item=5):

    print("\nApplying Single-Pass Filtering...\n")

    original_users = len(set(r["user_id"] for r in interactions))
    original_items = len(set(r["item_id"] for r in interactions))
    original_interactions = len(interactions)

    # ------------------------------------------------------
    # Count item frequencies ONCE
    # ------------------------------------------------------

    item_counts = Counter()

    for record in interactions:
        item_counts[record["item_id"]] += 1

    # ------------------------------------------------------
    # Build user sequences
    # ------------------------------------------------------

    user_sequences = defaultdict(list)

    for record in interactions:

        if item_counts[record["item_id"]] >= min_item:
            user_sequences[record["user_id"]].append(record)

    # ------------------------------------------------------
    # Remove users with fewer than min_user interactions
    # ------------------------------------------------------

    filtered = []

    for user, sequence in user_sequences.items():

        if len(sequence) >= min_user:
            filtered.extend(sequence)

    # ------------------------------------------------------
    # Final Statistics
    # ------------------------------------------------------

    final_users = len(set(r["user_id"] for r in filtered))
    final_items = len(set(r["item_id"] for r in filtered))
    final_interactions = len(filtered)

    print("=" * 65)
    print("FILTERING SUMMARY")
    print("=" * 65)

    print(f"Users        : {original_users:,} -> {final_users:,}")
    print(f"Items        : {original_items:,} -> {final_items:,}")
    print(f"Interactions : {original_interactions:,} -> {final_interactions:,}")

    print("\nRetention")
    print("-" * 30)
    print(f"Users        : {(final_users/original_users)*100:.2f}%")
    print(f"Items        : {(final_items/original_items)*100:.2f}%")
    print(f"Interactions : {(final_interactions/original_interactions)*100:.2f}%")

    return filtered


# ==========================================================
# Sort User Sequences
# ==========================================================

def sort_sequences(interactions):

    user_sequences = defaultdict(list)

    for record in interactions:
        user_sequences[record["user_id"]].append(record)

    final_dataset = []

    for user in user_sequences:

        user_sequences[user].sort(
            key=lambda x: x["timestamp"]
        )

        final_dataset.extend(user_sequences[user])

    return final_dataset


# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(interactions,
                 output_file):

    with open(output_file, "w", encoding="utf-8") as f:

        for record in interactions:

            json.dump(record, f)
            f.write("\n")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\nStarting Dataset Filtering...\n")

    for dataset_name, filename in CLEAN_DATASETS.items():

        print("\n" + "=" * 70)
        print(dataset_name.upper())
        print("=" * 70)

        processed_dir = os.path.join(
            DATASETS_ROOT,
            dataset_name,
            "processed"
        )

        input_file = os.path.join(
            processed_dir,
            filename
        )

        if not os.path.exists(input_file):

            print(f"Missing file:\n{input_file}")
            continue

        interactions = load_dataset(input_file)

        filtered = filter_dataset(
            interactions,
            MIN_USER_INTERACTIONS,
            MIN_ITEM_INTERACTIONS
        )

        filtered = sort_sequences(filtered)

        output_file = os.path.join(
            processed_dir,
            filename.replace("_clean", "_filtered")
        )

        save_dataset(
            filtered,
            output_file
        )

        print("\nSaved filtered dataset to:")
        print(output_file)

        print(f"\nFinal Users        : {len(set(r['user_id'] for r in filtered)):,}")
        print(f"Final Items        : {len(set(r['item_id'] for r in filtered)):,}")
        print(f"Final Interactions : {len(filtered):,}")

    print("\nFiltering complete.")