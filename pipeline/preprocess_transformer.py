import json
import os
from collections import defaultdict

# 1. Base Project Configuration
# Automatically resolves to C:\Users\deree\My Work\WITS\Honours\Project\Wits_Honours_Project\pipeline
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Goes up one level to Wits_Honours_Project
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Automatically resolves to C:\Users\deree\My Work\WITS\Honours\Project\Wits_Honours_Project\datasets
DATASETS_ROOT = os.path.join(BASE_DIR, "datasets")

# Uncomment the others when you are ready to process all of them
DOMAINS = [
    "Fashion", 
    # "Movies and TV Shows", 
    # "Video Games"
]

def load_clean_and_sort_interactions(file_path):
    user_histories = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            user_id = item.get("user_id")
            item_id = item.get("parent_asin") or item.get("asin")
            timestamp = item.get("timestamp")

            if not user_id or not item_id or timestamp is None:
                continue
            if not isinstance(user_id, str) or not isinstance(item_id, str):
                continue
            if isinstance(timestamp, (int, float)) and timestamp <= 0:
                continue

            user_histories[user_id].append((timestamp, item_id))

    clean_histories = {}
    for user_id, interactions in user_histories.items():
        interactions.sort(key=lambda x: x[0])
        
        deduped_interactions = []
        for ts, item_id in interactions:
            if not deduped_interactions or deduped_interactions[-1][1] != item_id:
                deduped_interactions.append((ts, item_id))
                
        clean_histories[user_id] = deduped_interactions

    return clean_histories

def apply_custom_core_filter(user_sequences, min_user_len=5, min_item_count=1):
    print(f"Executing iterative filter (Users >= {min_user_len}, Items >= {min_item_count})...")
    iteration = 0
    while True:
        iteration += 1
        start_user_count = len(user_sequences)

        item_counts = defaultdict(int)
        for seq in user_sequences.values():
            for _, item_id in seq:
                item_counts[item_id] += 1

        filtered_sequences = {}
        for user_id, seq in user_sequences.items():
            clean_seq = [(ts, item_id) for ts, item_id in seq if item_counts[item_id] >= min_item_count]
            if len(clean_seq) >= min_user_len:
                filtered_sequences[user_id] = clean_seq

        end_user_count = len(filtered_sequences)
        print(f"    Iteration {iteration}: Active Users {start_user_count} -> {end_user_count}")

        if start_user_count == end_user_count:
            print("Dataset successfully stabilized.")
            break
            
        user_sequences = filtered_sequences

    return user_sequences

if __name__ == "__main__":
    print("Starting Local RecBole Data Export...")

    for domain in DOMAINS:
        domain_dir = os.path.join(DATASETS_ROOT, domain)
        
        interaction_file = None
        if os.path.exists(domain_dir):
            for file in os.listdir(domain_dir):
                # Safely ignores meta_Amazon_Fashion.jsonl
                if file.endswith(".jsonl") and not file.startswith("meta_"):
                    interaction_file = os.path.join(domain_dir, file)
                    break

        if not interaction_file:
            print(f"[Error]: Could not find interaction JSONL inside {domain_dir}. Skipping...")
            continue

        try:
            print(f" -> Processing domain: {domain}")
            raw_histories = load_clean_and_sort_interactions(interaction_file)
            filtered_histories = apply_custom_core_filter(raw_histories, min_user_len=5, min_item_count=1)

            output_inter = os.path.join(domain_dir, f"{domain}.inter")
            total_rows = 0
            with open(output_inter, "w", encoding="utf-8") as fout:
                fout.write("user_id:token\titem_id:token\ttimestamp:float\n")
                for user_id, seq in filtered_histories.items():
                    for ts, item_id in seq:
                        fout.write(f"{user_id}\t{item_id}\t{ts}\n")
                        total_rows += 1

            print(f" Successfully exported {total_rows:,} rows to {domain}.inter!")

        except Exception as e:
            print(f"[Error] Failed to process domain {domain}: {e}")