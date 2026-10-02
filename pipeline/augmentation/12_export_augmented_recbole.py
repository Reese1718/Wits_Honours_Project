"""
12_export_augmented_recbole.py
    python pipeline/augmentation/12_export_augmented_recbole.py --dataset fashion
    python pipeline/augmentation/12_export_augmented_recbole.py --dataset video_games
    python pipeline/augmentation/12_export_augmented_recbole.py --dataset movies_tv
"""
import json
import os
import argparse
from numbers import Integral
import ijson
# ==========================================================
# Dataset Configuration
# ==========================================================
DATASETS = {
    "fashion": {
        "folder": "Fashion",
        "prefix": "Fashion",
    },
    "video_games": {
        "folder": "Video Games",
        "prefix": "Video_Games",
    },
    "movies_tv": {
        "folder": "Movies and TV Shows",
        "prefix": "Movies_TV",
    },
}
# ==========================================================
# Configuration
# ==========================================================
EXPECTED_AUGMENTATIONS = 3
# ==========================================================
# Arguments
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export augmented interaction histories "
            "to RecBole .inter format."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
        help=(
            "Dataset to process: "
            "fashion, video_games, or movies_tv"
        ),
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Optional custom input JSON file. "
            "If omitted, the default LLaMA augmented file is used."
        ),
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help=(
            "Optional custom output .inter file. "
            "If omitted, the default LLaMA augmented file is used."
        ),
    )
    parser.add_argument(
        "--selective",
        action="store_true",
        help=(
            "Indicate that the input is a selective augmentation "
            "dataset, where users may have 0 or 3 synthetic interactions."
        ),
    )
    return parser.parse_args()
# ==========================================================
# Helpers
# ==========================================================
def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()
def load_json(path):
    print("Loading:")
    print(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)