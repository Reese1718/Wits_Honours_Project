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

FILTERED_DATASETS = {
    "Fashion": "Fashion_filtered.jsonl",
    "Movies and TV Shows": "Movies_and_TV_Shows_filtered.jsonl",
    "Video Games": "Video_Games_filtered.jsonl"
}

dataset_results = []

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset(filepath):

    interactions = []

    with open(filepath, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                interactions.append(json.loads(line))

    return interactions


# ==========================================================
# Analyse Dataset
# ==========================================================

def analyse_dataset(dataset_name, interactions):

    user_counts = Counter()
    item_counts = Counter()

    for record in interactions:
        user_counts[record["user_id"]] += 1
        item_counts[record["item_id"]] += 1

    users = len(user_counts)
    items = len(item_counts)
    total_interactions = len(interactions)

    avg_user = total_interactions / users
    avg_item = total_interactions / items

    sparsity = 1 - (
        total_interactions /
        (users * items)
    )

    user_lengths = list(user_counts.values())
    item_lengths = list(item_counts.values())

    print("\n" + "=" * 70)
    print(dataset_name.upper())
    print("=" * 70)

    # GENERAL

    print("\nGENERAL")
    print("-" * 40)
    print(f"Users                 : {users:,}")
    print(f"Items                 : {items:,}")
    print(f"Interactions          : {total_interactions:,}")
    print(f"Sparsity              : {sparsity:.4%}")

    # USER STATS

    print("\nUSER STATISTICS")
    print("-" * 40)
    print(f"Average               : {statistics.mean(user_lengths):.2f}")
    print(f"Median                : {statistics.median(user_lengths)}")
    print(f"Minimum               : {min(user_lengths)}")
    print(f"Maximum               : {max(user_lengths)}")
    print(f"Std Dev               : {statistics.stdev(user_lengths):.2f}")

    # ITEM STATS

    print("\nITEM STATISTICS")
    print("-" * 40)
    print(f"Average               : {statistics.mean(item_lengths):.2f}")
    print(f"Median                : {statistics.median(item_lengths)}")
    print(f"Minimum               : {min(item_lengths)}")
    print(f"Maximum               : {max(item_lengths)}")
    print(f"Std Dev               : {statistics.stdev(item_lengths):.2f}")

    # USER PERCENTILES

    print("\nUSER PERCENTILES")
    print("-" * 40)

    user_lengths.sort()

    for p in [50, 75, 90, 95, 99]:
        index = min(int((p / 100) * len(user_lengths)), len(user_lengths) - 1)
        print(f"{p:>2}% : {user_lengths[index]}")

    # ITEM PERCENTILES

    print("\nITEM PERCENTILES")
    print("-" * 40)

    item_lengths.sort()

    for p in [50, 75, 90, 95, 99]:
        index = min(int((p / 100) * len(item_lengths)), len(item_lengths) - 1)
        print(f"{p:>2}% : {item_lengths[index]}")

    # LONG TAIL

    print("\nLONG TAIL ANALYSIS")
    print("-" * 40)

    print(f"Items <=5 interactions  : {sum(x <= 5 for x in item_lengths)/items:.2%}")
    print(f"Items <=10 interactions : {sum(x <= 10 for x in item_lengths)/items:.2%}")
    print(f"Items >50 interactions  : {sum(x > 50 for x in item_lengths)/items:.2%}")
    print(f"Items >100 interactions : {sum(x > 100 for x in item_lengths)/items:.2%}")

    # USER HISTORY

    print("\nUSER HISTORY ANALYSIS")
    print("-" * 40)

    print(f"Users <=5 interactions  : {sum(x <= 5 for x in user_lengths)/users:.2%}")
    print(f"Users <=10 interactions : {sum(x <= 10 for x in user_lengths)/users:.2%}")
    print(f"Users >20 interactions  : {sum(x > 20 for x in user_lengths)/users:.2%}")

    # USER DISTRIBUTION

    print("\nUSER INTERACTION DISTRIBUTION")
    print("-" * 40)

    user_distribution = Counter(user_lengths)

    for i in range(1, 21):
        count = user_distribution.get(i, 0)
        pct = count / users * 100
        print(f"{i:>2} interactions : {count:>8,} users ({pct:6.2f}%)")

    over20 = sum(v for k, v in user_distribution.items() if k > 20)
    print(f">20 interactions : {over20:>8,} users ({over20/users*100:6.2f}%)")

    # ITEM DISTRIBUTION

    print("\nITEM POPULARITY DISTRIBUTION")
    print("-" * 40)

    item_distribution = Counter(item_lengths)

    for i in range(1, 21):
        count = item_distribution.get(i, 0)
        pct = count / items * 100
        print(f"{i:>2} interactions : {count:>8,} items ({pct:6.2f}%)")

    over20 = sum(v for k, v in item_distribution.items() if k > 20)
    print(f">20 interactions : {over20:>8,} items ({over20/items*100:6.2f}%)")

    dataset_results.append({
        "Dataset": dataset_name,
        "Users": users,
        "Items": items,
        "Interactions": total_interactions,
        "Avg/User": round(avg_user, 2),
        "Avg/Item": round(avg_item, 2),
        "Sparsity": sparsity
    })


# ==========================================================
# Final Comparison Table
# ==========================================================

def print_comparison():

    print("\n")
    print("=" * 100)
    print("FINAL DATASET COMPARISON")
    print("=" * 100)

    print(f"{'Dataset':25}"
          f"{'Users':>12}"
          f"{'Items':>12}"
          f"{'Interactions':>18}"
          f"{'Avg/User':>12}"
          f"{'Avg/Item':>12}"
          f"{'Sparsity':>14}")

    print("-" * 100)

    for row in dataset_results:

        print(
            f"{row['Dataset']:25}"
            f"{row['Users']:>12,}"
            f"{row['Items']:>12,}"
            f"{row['Interactions']:>18,}"
            f"{row['Avg/User']:>12.2f}"
            f"{row['Avg/Item']:>12.2f}"
            f"{row['Sparsity']:>13.4%}"
        )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\nGenerating Dataset Comparison Report...")

    for dataset_name, filename in FILTERED_DATASETS.items():

        processed_dir = os.path.join(
            DATASETS_ROOT,
            dataset_name,
            "processed"
        )

        filepath = os.path.join(processed_dir, filename)

        if not os.path.exists(filepath):

            print(f"Missing dataset:\n{filepath}")
            continue

        interactions = load_dataset(filepath)

        analyse_dataset(dataset_name, interactions)

    print_comparison()

    print("\nDataset comparison complete.")