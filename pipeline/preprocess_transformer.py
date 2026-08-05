import json
import os
import statistics
from collections import defaultdict

# 1. Base Project Configuration
# Automatically resolves to C:\Users\deree\My Work\WITS\Honours\Project\Wits_Honours_Project\pipeline
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

        deduped = []

        for ts, item in interactions:

            # Remove ONLY identical duplicate events
            if deduped:
                last_ts, last_item = deduped[-1]
                if ts == last_ts and item == last_item:
                    continue

            deduped.append((ts, item))

        clean_histories[user_id] = deduped

    return clean_histories

def apply_custom_core_filter(user_sequences,
                             min_user_len=5,
                             min_item_count=5):

    print(f"Applying single-pass filter (Users >= {min_user_len}, Items >= {min_item_count})...")

    # Count item frequencies on the raw dataset
    item_counts = defaultdict(int)

    for seq in user_sequences.values():
        for _, item in seq:
            item_counts[item] += 1

    filtered_sequences = {}

    for user, seq in user_sequences.items():

        new_seq = [
            (ts, item)
            for ts, item in seq
            if item_counts[item] >= min_item_count
        ]

        if len(new_seq) >= min_user_len:
            filtered_sequences[user] = new_seq

    print(f"Users remaining: {len(filtered_sequences):,}")

    return filtered_sequences

def print_dataset_statistics(user_sequences, domain_name):
    total_users = len(user_sequences)
    unique_items = set()
    seq_lengths = []
    
    for seq in user_sequences.values():
        seq_lengths.append(len(seq))
        for _, item_id in seq:
            unique_items.add(item_id)
            
    total_items = len(unique_items)
    total_interactions = sum(seq_lengths)
    
    if total_users == 0 or total_items == 0:
        print(f"\n[Warning] Dataset {domain_name} is completely empty after filtering.")
        return
        
    avg_seq_len = total_interactions / total_users
    median_seq_len = statistics.median(seq_lengths)
    max_seq_len = max(seq_lengths)
    min_seq_len = min(seq_lengths)
    
    sparsity = 1.0 - (total_interactions / (total_users * total_items))
    avg_item_actions = total_interactions / total_items
    
    print("\n" + "="*40)
    print(f"{domain_name} Dataset Summary (After 5-Core)")
    print("="*40)
    print(f"Users:                  {total_users:,}")
    print(f"Items:                  {total_items:,}")
    print(f"Interactions:           {total_interactions:,}")
    print(f"Avg sequence length:    {avg_seq_len:.2f}")
    print(f"Median sequence length: {median_seq_len}")
    print(f"Min sequence length:    {min_seq_len}")
    print(f"Max sequence length:    {max_seq_len}")
    print(f"Dataset sparsity:       {sparsity:.4%}")
    print(f"Avg actions/item:       {avg_item_actions:.2f}")
    print("="*40 + "\n")

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
            
            # EXECUTING TRUE 5-CORE FILTER HERE
            filtered_histories = apply_custom_core_filter(raw_histories, min_user_len=5, min_item_count=5)
            
            print_dataset_statistics(filtered_histories, domain)

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