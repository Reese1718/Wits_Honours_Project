import json
import os
from collections import defaultdict

# ==========================================================
# Project Configuration
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASETS_ROOT = os.path.join(BASE_DIR, "datasets")

# Raw interaction datasets
RAW_DATASETS = {
    "Fashion": "Amazon_Fashion.jsonl",
    "Movies and TV Shows": "Movies_and_TV.jsonl",
    "Video Games": "Video_Games.jsonl"
}


# ==========================================================
# Clean Dataset
# ==========================================================

def clean_dataset(input_file):

    user_histories = defaultdict(list)

    stats = {
        "records_read": 0,
        "malformed_json": 0,
        "missing_fields": 0,
        "invalid_timestamp": 0,
        "duplicate_interactions": 0,
        "records_written": 0
    }

    with open(input_file, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                stats["malformed_json"] += 1
                continue

            stats["records_read"] += 1

            user = record.get("user_id")
            item = record.get("parent_asin") or record.get("asin")
            timestamp = record.get("timestamp")

            if user is None or item is None or timestamp is None:
                stats["missing_fields"] += 1
                continue

            if not isinstance(timestamp, (int, float)):
                stats["invalid_timestamp"] += 1
                continue

            if timestamp <= 0:
                stats["invalid_timestamp"] += 1
                continue

            user_histories[user].append((timestamp, item))

    cleaned = []

    for user, interactions in user_histories.items():

        interactions.sort(key=lambda x: x[0])

        seen = set()

        for timestamp, item in interactions:

            interaction = (item, timestamp)

            if interaction in seen:
                stats["duplicate_interactions"] += 1
                continue

            seen.add(interaction)

            cleaned.append({
                "user_id": user,
                "item_id": item,
                "timestamp": timestamp
            })

    stats["records_written"] = len(cleaned)

    return cleaned, stats


# ==========================================================
# Save JSONL
# ==========================================================

def save_jsonl(records, output_file):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:

        for record in records:
            json.dump(record, f)
            f.write("\n")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\nStarting Dataset Cleaning...\n")

    for dataset_name, raw_filename in RAW_DATASETS.items():

        dataset_dir = os.path.join(DATASETS_ROOT, dataset_name)

        input_file = os.path.join(dataset_dir, raw_filename)

        if not os.path.exists(input_file):

            print(f"[WARNING] Could not find {raw_filename}")
            continue

        print("=" * 70)
        print(f"Cleaning {dataset_name}")
        print("=" * 70)

        cleaned_records, stats = clean_dataset(input_file)

        processed_dir = os.path.join(dataset_dir, "processed")

        output_file = os.path.join(
            processed_dir,
            f"{dataset_name.replace(' ', '_')}_clean.jsonl"
        )

        save_jsonl(cleaned_records, output_file)

        print(f"Output File:")
        print(output_file)
        print()

        print("Cleaning Summary")
        print("-" * 40)

        for key, value in stats.items():
            print(f"{key:<30}{value:,}")

        print()