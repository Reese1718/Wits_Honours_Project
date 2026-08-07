import json
import os
from collections import defaultdict

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

# ==========================================================
# Load Filtered Dataset
# ==========================================================

def load_dataset(filepath):

    interactions = []

    with open(filepath, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():

                interactions.append(json.loads(line))

    return interactions


# ==========================================================
# Sort User Histories
# ==========================================================

def sort_user_histories(interactions):

    user_histories = defaultdict(list)

    for interaction in interactions:

        user_histories[
            interaction["user_id"]
        ].append(interaction)

    sorted_interactions = []

    for user in user_histories:

        user_histories[user].sort(
            key=lambda x: x["timestamp"]
        )

        sorted_interactions.extend(
            user_histories[user]
        )

    return sorted_interactions


# ==========================================================
# Export RecBole File
# ==========================================================

def export_recbole(interactions, output_file):

    with open(output_file, "w", encoding="utf-8") as fout:

        fout.write(
            "user_id:token\titem_id:token\ttimestamp:float\n"
        )

        for interaction in interactions:

            fout.write(
                f"{interaction['user_id']}\t"
                f"{interaction['item_id']}\t"
                f"{interaction['timestamp']}\n"
            )

    print(f"Exported {len(interactions):,} interactions.")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\nExporting RecBole Datasets...\n")

    for dataset_name, filename in FILTERED_DATASETS.items():

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

            print(f"Missing dataset:\n{input_file}")
            continue

        print("=" * 70)
        print(dataset_name.upper())
        print("=" * 70)

        interactions = load_dataset(input_file)

        interactions = sort_user_histories(
            interactions
        )

        output_file = os.path.join(
            processed_dir,
            f"{dataset_name}.inter"
        )

        export_recbole(
            interactions,
            output_file
        )

        print(f"Saved to:\n{output_file}\n")

    print("RecBole export complete.")