import json
import os
import statistics
from collections import Counter

# ==========================================================
# Project Configuration
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASETS_ROOT = os.path.join(BASE_DIR, "datasets")

# Clean datasets produced by 02_clean_dataset.py
CLEAN_DATASETS = {
    "Fashion": "Fashion_clean.jsonl",
    "Movies and TV Shows": "Movies_and_TV_Shows_clean.jsonl",
    "Video Games": "Video_Games_clean.jsonl"
}


# ==========================================================
# Load Clean Dataset
# ==========================================================

def load_dataset(file_path):

    user_counter = Counter()
    item_counter = Counter()

    total_interactions = 0

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            record = json.loads(line)

            user = record["user_id"]
            item = record["item_id"]

            user_counter[user] += 1
            item_counter[item] += 1
            total_interactions += 1

    return user_counter, item_counter, total_interactions


# ==========================================================
# Dataset Statistics
# ==========================================================

def print_statistics(dataset_name,
                     user_counter,
                     item_counter,
                     interactions):

    users = len(user_counter)
    items = len(item_counter)

    user_lengths = list(user_counter.values())
    item_popularity = list(item_counter.values())

    sparsity = 1 - (interactions / (users * items))

    print("\n" + "=" * 70)
    print(dataset_name.upper())
    print("=" * 70)

    print("\nGENERAL")
    print("-" * 40)
    print(f"Users                 : {users:,}")
    print(f"Items                 : {items:,}")
    print(f"Interactions          : {interactions:,}")
    print(f"Sparsity              : {sparsity:.4%}")

    print("\nUSER STATISTICS")
    print("-" * 40)
    print(f"Average               : {statistics.mean(user_lengths):.2f}")
    print(f"Median                : {statistics.median(user_lengths)}")
    print(f"Minimum               : {min(user_lengths)}")
    print(f"Maximum               : {max(user_lengths)}")
    print(f"Std Dev               : {statistics.stdev(user_lengths):.2f}")

    print("\nITEM STATISTICS")
    print("-" * 40)
    print(f"Average               : {statistics.mean(item_popularity):.2f}")
    print(f"Median                : {statistics.median(item_popularity)}")
    print(f"Minimum               : {min(item_popularity)}")
    print(f"Maximum               : {max(item_popularity)}")
    print(f"Std Dev               : {statistics.stdev(item_popularity):.2f}")

    print("\nUSER INTERACTION DISTRIBUTION")
    print("-" * 40)

    for i in range(1, 11):

        count = sum(v == i for v in user_lengths)

        print(f"{i:>2} interactions : {count:,}")

    print(f">10 interactions: {sum(v > 10 for v in user_lengths):,}")

    print("\nITEM POPULARITY DISTRIBUTION")
    print("-" * 40)

    for i in range(1, 11):

        count = sum(v == i for v in item_popularity)

        print(f"{i:>2} interactions : {count:,}")

    print(f">10 interactions: {sum(v > 10 for v in item_popularity):,}")

    print("\nSEQUENCE LENGTH PERCENTILES")
    print("-" * 40)

    user_lengths.sort()

    n = len(user_lengths)

    print(f"90th Percentile       : {user_lengths[int(0.90*n)]}")
    print(f"95th Percentile       : {user_lengths[int(0.95*n)]}")
    print(f"99th Percentile       : {user_lengths[int(0.99*n)]}")

    print("=" * 70)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\nGenerating Dataset Statistics...\n")

    for dataset, filename in CLEAN_DATASETS.items():

        processed_dir = os.path.join(
            DATASETS_ROOT,
            dataset,
            "processed"
        )

        dataset_file = os.path.join(processed_dir, filename)

        if not os.path.exists(dataset_file):

            print(f"[WARNING] Missing cleaned dataset:")
            print(dataset_file)
            continue

        users, items, interactions = load_dataset(dataset_file)

        print_statistics(
            dataset,
            users,
            items,
            interactions
        )