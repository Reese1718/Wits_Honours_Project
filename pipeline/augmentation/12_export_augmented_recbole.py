#!/usr/bin/env python3
"""Export augmented user histories to RecBole .inter format.

Selective Fashion examples:
  python pipeline/augmentation/12_export_augmented_recbole.py --dataset fashion --model qwen --selective
  python pipeline/augmentation/12_export_augmented_recbole.py --dataset fashion --model llama --selective
"""

import argparse
import os
import tempfile
from pathlib import Path

import ijson


DATASETS = {
    "fashion": {"folder": "Fashion", "prefix": "Fashion"},
    "video_games": {"folder": "Video Games", "prefix": "Video_Games"},
    "movies_tv": {"folder": "Movies and TV Shows", "prefix": "Movies_TV"},
}
EXPECTED_AUGMENTATIONS = 3
VALID_SOURCES = {
    "real_train",
    "llm_synthetic",
    "real_validation",
    "real_test",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export augmented histories to RecBole .inter format."
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--model",
        choices=("qwen", "llama"),
        default="qwen",
        help="LLM used for augmentation (default: qwen).",
    )
    parser.add_argument(
        "--selective",
        action="store_true",
        help="Allow users to have either 0 or 3 synthetic interactions.",
    )
    parser.add_argument(
        "--input-file",
        help="Custom input JSON. Relative paths use the project root.",
    )
    parser.add_argument(
        "--output-file",
        help="Custom output .inter file. Relative paths use the project root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def clean_text(value):
    return "" if value is None else str(value).strip()


def project_root():
    # The script must be stored in pipeline/augmentation/.
    return Path(__file__).resolve().parents[2]


def resolve_path(value, root):
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def default_paths(root, dataset_key, model, selective):
    config = DATASETS[dataset_key]
    prefix = config["prefix"]
    export_dir = root / "datasets" / config["folder"] / "exports"

    if selective:
        stem = f"{prefix}_selective_{model}_augmented"
    elif model == "qwen":
        stem = f"{prefix}_augmented"
    else:
        stem = f"{prefix}_llama_augmented"

    return (
        export_dir / f"{stem}_user_histories.json",
        export_dir / f"{stem}.inter",
    )


def history_prefix(path):
    """Support both {user_id: [...]} and {users: {user_id: [...]}}."""
    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            if prefix == "" and event == "map_key":
                return "users" if value == "users" else ""
            if prefix == "" and event == "start_array":
                raise RuntimeError(
                    "Input must be an object mapping user IDs to histories."
                )
    raise RuntimeError("Input JSON is empty or is not an object.")


def iter_histories(path):
    prefix = history_prefix(path)
    with path.open("rb") as handle:
        yield from ijson.kvitems(handle, prefix)


def normalise_position(value, user_id):
    if isinstance(value, bool) or value is None:
        raise RuntimeError(f"{user_id}: invalid sequence_position {value!r}.")
    try:
        position = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"{user_id}: invalid sequence_position {value!r}."
        ) from exc

    valid = (
        value.strip() == str(position)
        if isinstance(value, str)
        else value == position
    )
    if not valid or position < 1:
        raise RuntimeError(f"{user_id}: invalid sequence_position {value!r}.")
    return position


def export_augmented_recbole():
    args = parse_args()
    root = project_root()
    default_input, default_output = default_paths(
        root, args.dataset, args.model, args.selective
    )
    input_path = (
        resolve_path(args.input_file, root)
        if args.input_file
        else default_input
    )
    output_path = (
        resolve_path(args.output_file, root)
        if args.output_file
        else default_output
    )

    print("=" * 70)
    print("EXPORTING AUGMENTED HISTORIES TO RECBOLE")
    print("=" * 70)
    print(f"Dataset:   {DATASETS[args.dataset]['folder']}")
    print(f"Model:     {args.model}")
    print(f"Selective: {args.selective}")
    print(f"Input:     {input_path}")
    print(f"Output:    {output_path}")

    if not input_path.is_file():
        raise FileNotFoundError(f"Required input file not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ.")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "users": 0,
        "real_train": 0,
        "llm_synthetic": 0,
        "real_validation": 0,
        "real_test": 0,
        "interactions": 0,
        "synthetic_users": 0,
    }
    seen_users = set()
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        ) as output:
            temporary_path = Path(output.name)
            output.write("user_id:token\titem_id:token\ttimestamp:float\n")

            for raw_user_id, sequence in iter_histories(input_path):
                user_id = clean_text(raw_user_id)
                if not user_id:
                    raise RuntimeError("Encountered an empty user ID.")
                if user_id in seen_users:
                    raise RuntimeError(f"Duplicate user: {user_id}")
                seen_users.add(user_id)

                if not isinstance(sequence, list) or not sequence:
                    raise RuntimeError(
                        f"{user_id}: expected a non-empty interaction list."
                    )
                if any(not isinstance(item, dict) for item in sequence):
                    raise RuntimeError(
                        f"{user_id}: every interaction must be an object."
                    )

                prepared = [
                    (normalise_position(item.get("sequence_position"), user_id), item)
                    for item in sequence
                ]
                positions = [position for position, _ in prepared]
                if len(set(positions)) != len(positions):
                    raise RuntimeError(f"{user_id}: duplicate sequence positions.")
                prepared.sort(key=lambda pair: pair[0])

                sorted_positions = [position for position, _ in prepared]
                if (
                    not args.selective
                    and sorted_positions != list(range(1, len(prepared) + 1))
                ):
                    raise RuntimeError(
                        f"{user_id}: non-selective positions must be consecutive."
                    )

                sources = [
                    clean_text(item.get("interaction_source"))
                    for _, item in prepared
                ]
                unknown = set(sources) - VALID_SOURCES
                if unknown:
                    raise RuntimeError(
                        f"{user_id}: unknown interaction source(s): {sorted(unknown)}"
                    )

                user_synthetic = sources.count("llm_synthetic")
                allowed = {0, EXPECTED_AUGMENTATIONS} if args.selective else {
                    EXPECTED_AUGMENTATIONS
                }
                if user_synthetic not in allowed:
                    raise RuntimeError(
                        f"{user_id}: expected synthetic count in {sorted(allowed)}, "
                        f"found {user_synthetic}."
                    )
                if sources.count("real_validation") != 1:
                    raise RuntimeError(
                        f"{user_id}: expected exactly one validation interaction."
                    )
                if sources.count("real_test") != 1:
                    raise RuntimeError(
                        f"{user_id}: expected exactly one test interaction."
                    )
                if len(prepared) < 3:
                    raise RuntimeError(f"{user_id}: fewer than 3 interactions.")
                if sources[-2:] != ["real_validation", "real_test"]:
                    raise RuntimeError(
                        f"{user_id}: validation/test must be the final two interactions."
                    )

                for (position, item), source in zip(prepared, sources):
                    item_id = clean_text(item.get("asin"))
                    if not item_id:
                        raise RuntimeError(
                            f"{user_id}: missing ASIN at position {position}."
                        )
                    output.write(f"{user_id}\t{item_id}\t{float(position)}\n")
                    counts[source] += 1
                    counts["interactions"] += 1

                counts["users"] += 1
                if user_synthetic == EXPECTED_AUGMENTATIONS:
                    counts["synthetic_users"] += 1

        if counts["users"] == 0:
            raise RuntimeError("No users were found in the input file.")
        if counts["real_validation"] != counts["users"]:
            raise RuntimeError("Global validation count does not match user count.")
        if counts["real_test"] != counts["users"]:
            raise RuntimeError("Global test count does not match user count.")
        if counts["llm_synthetic"] != (
            counts["synthetic_users"] * EXPECTED_AUGMENTATIONS
        ):
            raise RuntimeError("Global synthetic count reconciliation failed.")

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print("\nExport validation passed.")
    print(f"Users:                   {counts['users']:,}")
    print(f"Real train interactions: {counts['real_train']:,}")
    print(f"Synthetic interactions:  {counts['llm_synthetic']:,}")
    print(f"Validation interactions: {counts['real_validation']:,}")
    print(f"Test interactions:       {counts['real_test']:,}")
    print(f"Total interactions:      {counts['interactions']:,}")
    if args.selective:
        print(f"Users augmented:          {counts['synthetic_users']:,}")
        print(
            f"Users not augmented:      "
            f"{counts['users'] - counts['synthetic_users']:,}"
        )
    print(f"\nCreated: {output_path}")


if __name__ == "__main__":
    export_augmented_recbole()
