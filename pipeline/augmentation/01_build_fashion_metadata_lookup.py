import os
import json


# ==========================================================
# Paths
# ==========================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(PIPELINE_DIR)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "datasets",
    "Fashion"
)

PROCESSED_DIR = os.path.join(
    DATASET_DIR,
    "processed"
)

EXPORT_DIR = os.path.join(
    DATASET_DIR,
    "exports"
)

FILTERED_FILE = os.path.join(
    PROCESSED_DIR,
    "Fashion_filtered.jsonl"
)

METADATA_FILE = os.path.join(
    DATASET_DIR,
    "meta_Amazon_Fashion.jsonl"
)

OUTPUT_FILE = os.path.join(
    EXPORT_DIR,
    "Fashion_metadata_lookup.json"
)


# ==========================================================
# Create output directory
# ==========================================================

os.makedirs(
    EXPORT_DIR,
    exist_ok=True
)


# ==========================================================
# Load filtered ASINs
# ==========================================================

print("=" * 70)
print("BUILDING FASHION METADATA LOOKUP")
print("=" * 70)

print("\nLoading filtered interaction dataset:")
print(FILTERED_FILE)

filtered_asins = set()

with open(
    FILTERED_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        if "item_id" in record:
            asin = record["item_id"]

        elif "asin" in record:
            asin = record["asin"]

        elif "parent_asin" in record:
            asin = record["parent_asin"]

        else:
            continue

        filtered_asins.add(
            str(asin)
        )

print(
    "\nUnique filtered ASINs:",
    len(filtered_asins)
)


# ==========================================================
# Load metadata
# ==========================================================

print("\nLoading Amazon Fashion metadata:")
print(METADATA_FILE)

metadata_lookup = {}

metadata_records = 0
matched_records = 0

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        metadata_records += 1

        asin = (
            record.get("parent_asin")
            or record.get("asin")
        )

        if asin is None:
            continue

        asin = str(asin)

        # Only keep products used in the filtered 5-core dataset
        if asin not in filtered_asins:
            continue

        metadata_lookup[asin] = {
            "asin": asin,
            "title": record.get("title"),
            "main_category": record.get("main_category"),
            "categories": record.get("categories"),
            "store": record.get("store"),
            "details": record.get("details"),
            "description": record.get("description"),
            "features": record.get("features")
        }

        matched_records += 1


# ==========================================================
# Save lookup
# ==========================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata_lookup,
        f,
        indent=2,
        ensure_ascii=False
    )


# ==========================================================
# Validation
# ==========================================================

matched_asins = set(
    metadata_lookup.keys()
)

missing_asins = (
    filtered_asins
    - matched_asins
)

print("\n" + "=" * 70)
print("METADATA LOOKUP COMPLETE")
print("=" * 70)

print(
    "\nMetadata records scanned:",
    metadata_records
)

print(
    "Filtered ASINs:",
    len(filtered_asins)
)

print(
    "Matched ASINs:",
    len(matched_asins)
)

print(
    "Missing ASINs:",
    len(missing_asins)
)

print(
    "\nOutput:",
    OUTPUT_FILE
)


# ==========================================================
# Show examples
# ==========================================================

print("\nExample metadata records:")
print("-" * 70)

for i, asin in enumerate(
    list(metadata_lookup.keys())[:5]
):

    print(
        json.dumps(
            metadata_lookup[asin],
            indent=2,
            ensure_ascii=False
        )
    )

    if i >= 4:
        break


# ==========================================================
# Show missing examples
# ==========================================================

if missing_asins:

    print("\nExample missing ASINs:")
    print("-" * 70)

    for asin in list(missing_asins)[:10]:
        print(asin)

else:

    print(
        "\nAll filtered ASINs have corresponding metadata."
    )

print("\n" + "=" * 70)