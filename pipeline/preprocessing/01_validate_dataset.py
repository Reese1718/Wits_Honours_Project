import json
import os
import statistics
from collections import defaultdict
from datetime import datetime

# ==========================================================
# Project Configuration
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASETS_ROOT = os.path.join(BASE_DIR, "datasets")


# ==========================================================
# Validation Function
# ==========================================================

def validate_dataset(jsonl_path):
    total_reviews = 0

    missing_user = 0
    missing_item = 0
    missing_timestamp = 0
    malformed_json = 0

    users = defaultdict(int)
    items = defaultdict(int)

    duplicate_checker = set()
    duplicate_interactions = 0

    min_timestamp = None
    max_timestamp = None

    with open(jsonl_path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                malformed_json += 1
                continue

            total_reviews += 1

            user = review.get("user_id")
            item = review.get("parent_asin") or review.get("asin")
            timestamp = review.get("timestamp")

            # -----------------------------
            # Missing values
            # -----------------------------

            if not user:
                missing_user += 1

            if not item:
                missing_item += 1

            if timestamp is None:
                missing_timestamp += 1

            if not user or not item or timestamp is None:
                continue

            # -----------------------------
            # User & Item Counts
            # -----------------------------

            users[user] += 1
            items[item] += 1

            # -----------------------------
            # Duplicate Interaction Check
            # -----------------------------

            interaction = (user, item, timestamp)

            if interaction in duplicate_checker:
                duplicate_interactions += 1
            else:
                duplicate_checker.add(interaction)

            # -----------------------------
            # Timestamp Range
            # -----------------------------

            if min_timestamp is None or timestamp < min_timestamp:
                min_timestamp = timestamp

            if max_timestamp is None or timestamp > max_timestamp:
                max_timestamp = timestamp

    user_lengths = list(users.values())
    item_lengths = list(items.values())

    report = {

        "total_reviews": total_reviews,

        "unique_users": len(users),

        "unique_items": len(items),

        "missing_user_ids": missing_user,

        "missing_item_ids": missing_item,

        "missing_timestamps": missing_timestamp,

        "malformed_json_records": malformed_json,

        "duplicate_interactions": duplicate_interactions,

        "earliest_timestamp": min_timestamp,

        "latest_timestamp": max_timestamp,

        "average_interactions_per_user":
            round(statistics.mean(user_lengths), 2) if user_lengths else 0,

        "median_interactions_per_user":
            statistics.median(user_lengths) if user_lengths else 0,

        "max_interactions_per_user":
            max(user_lengths) if user_lengths else 0,

        "average_interactions_per_item":
            round(statistics.mean(item_lengths), 2) if item_lengths else 0,

        "median_interactions_per_item":
            statistics.median(item_lengths) if item_lengths else 0,

        "max_interactions_per_item":
            max(item_lengths) if item_lengths else 0

    }

    return report


# ==========================================================
# Pretty Print Report
# ==========================================================

def print_report(dataset_name, report):

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} DATASET VALIDATION")
    print("=" * 70)

    print(f"Total Reviews               : {report['total_reviews']:,}")
    print(f"Unique Users               : {report['unique_users']:,}")
    print(f"Unique Items               : {report['unique_items']:,}")

    print()

    print(f"Missing User IDs           : {report['missing_user_ids']:,}")
    print(f"Missing Item IDs           : {report['missing_item_ids']:,}")
    print(f"Missing Timestamps         : {report['missing_timestamps']:,}")
    print(f"Malformed JSON Records     : {report['malformed_json_records']:,}")
    print(f"Duplicate Interactions     : {report['duplicate_interactions']:,}")

    print()

    if report["earliest_timestamp"] is not None:

        earliest = datetime.fromtimestamp(
            report["earliest_timestamp"] / 1000
        )

        latest = datetime.fromtimestamp(
            report["latest_timestamp"] / 1000
        )

        print(f"Earliest Review            : {earliest}")
        print(f"Latest Review              : {latest}")

    print()

    print(f"Average Interactions/User  : {report['average_interactions_per_user']}")
    print(f"Median Interactions/User   : {report['median_interactions_per_user']}")
    print(f"Maximum Interactions/User  : {report['max_interactions_per_user']}")

    print()

    print(f"Average Interactions/Item  : {report['average_interactions_per_item']}")
    print(f"Median Interactions/Item   : {report['median_interactions_per_item']}")
    print(f"Maximum Interactions/Item  : {report['max_interactions_per_item']}")

    print("=" * 70)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("Starting Dataset Validation...\n")

    if not os.path.exists(DATASETS_ROOT):
        print(f"[ERROR] Datasets folder not found:\n{DATASETS_ROOT}")
        exit()

    dataset_dirs = sorted([
        d for d in os.listdir(DATASETS_ROOT)
        if os.path.isdir(os.path.join(DATASETS_ROOT, d))
    ])

    if len(dataset_dirs) == 0:
        print("[ERROR] No dataset folders found.")
        exit()

    for dataset in dataset_dirs:

        dataset_dir = os.path.join(DATASETS_ROOT, dataset)

        interaction_file = None

        for file in os.listdir(dataset_dir):

            if file.endswith(".jsonl") and not file.startswith("meta_"):
                interaction_file = os.path.join(dataset_dir, file)
                break

        if interaction_file is None:

            print(f"[WARNING] No interaction dataset found for '{dataset}'.")
            continue

        report = validate_dataset(interaction_file)

        print_report(dataset, report)

        report_file = os.path.join(
            dataset_dir,
            "validation_report.json"
        )

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        print(f"\nValidation report saved to:")
        print(report_file)