"""
Prepare original user histories for selective GAN augmentation.

Input:
    datasets/<dataset>/exports/<prefix>_user_histories.json

Output:
    datasets/<dataset>/processed/gan/
        train_examples.jsonl
        dev_examples.jsonl
        users.jsonl
        item_catalogue.json
        preparation_report.json

Split policy:
    Original history: [training ..., validation, test]

    The last TRAINING item is used as an internal development target.
    Official validation and test items are never used for GAN training
    or checkpoint selection.

Eligibility:
    Original history length <= threshold, including the final two items.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import ijson


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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--max-history",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-examples-per-user",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--order-field",
        default="timestamp",
        help="Use timestamp or sequence_position to order histories.",
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if min(
        args.threshold,
        args.max_history,
        args.max_examples_per_user,
    ) < 1:
        parser.error("Threshold and length limits must be positive.")

    return args


def write_jsonl(file, record):
    file.write(
        json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def file_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def validate_and_sort(user_id, history, order_field):
    if not isinstance(history, list) or len(history) < 5:
        raise ValueError(
            f"{user_id}: expected at least 5 original interactions."
        )

    for interaction in history:
        if not isinstance(interaction, dict):
            raise ValueError(
                f"{user_id}: interaction must be a dictionary."
            )

        asin = interaction.get("asin")

        if not isinstance(asin, str) or not asin.strip():
            raise ValueError(
                f"{user_id}: missing or invalid asin."
            )

        source = str(
            interaction.get("interaction_source", "")
        ).lower()

        if "synthetic" in source:
            raise ValueError(
                f"{user_id}: input contains synthetic interactions. "
                "Use the original user-history file."
            )

        order = interaction.get(order_field)

        if (
            isinstance(order, bool)
            or not isinstance(order, (int, float))
            or not math.isfinite(order)
        ):
            raise ValueError(
                f"{user_id}: missing or invalid {order_field}."
            )

    # Python sorting is stable: timestamp ties retain source order.
    history = sorted(
        history,
        key=lambda interaction: interaction[order_field],
    )

    sources = [
        interaction.get("interaction_source")
        for interaction in history
    ]

    if "real_validation" in sources or "real_test" in sources:
        if (
            sources.count("real_validation") != 1
            or sources.count("real_test") != 1
            or sources[-2:] != ["real_validation", "real_test"]
        ):
            raise ValueError(
                f"{user_id}: existing split markers conflict "
                "with the last-two temporal split."
            )

    return history


def main():
    args = parse_args()
    dataset = DATASETS[args.dataset]

    project_dir = Path(__file__).resolve().parents[2]

    dataset_dir = (
        project_dir
        / "datasets"
        / dataset["folder"]
    )

    input_file = args.input_file or (
        dataset_dir
        / "exports"
        / f"{dataset['prefix']}_user_histories.json"
    )

    output_dir = dataset_dir / "processed" / "gan"

    if not input_file.is_file():
        raise FileNotFoundError(input_file)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {output_dir}\n"
            "Move the previous output before running preparation again."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": dataset["folder"],
        "input_file": str(input_file),
        "threshold": args.threshold,
        "eligibility_basis": "original total history length",
        "max_history": args.max_history,
        "max_examples_per_user": args.max_examples_per_user,
        "order_field": args.order_field,
        "users": 0,
        "eligible_users": 0,
        "original_interactions": 0,
        "real_training_interactions": 0,
        "training_examples": 0,
        "internal_dev_examples": 0,
        "repeated_training_targets": 0,
        "repeated_dev_targets": 0,
        "status": "INCOMPLETE",
    }

    report_path = output_dir / "preparation_report.json"

    # An incomplete report makes interrupted preparation detectable.
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    seen_users = set()
    training_items = set()

    with (
        input_file.open("rb") as source_file,
        (output_dir / "train_examples.jsonl").open(
            "w", encoding="utf-8"
        ) as train_file,
        (output_dir / "dev_examples.jsonl").open(
            "w", encoding="utf-8"
        ) as dev_file,
        (output_dir / "users.jsonl").open(
            "w", encoding="utf-8"
        ) as users_file,
    ):
        user_histories = ijson.kvitems(
            source_file,
            "",
            use_float=True,
        )

        for user_id, history in user_histories:
            if not user_id or user_id in seen_users:
                raise ValueError(
                    f"Empty or duplicate user ID: {user_id!r}"
                )

            seen_users.add(user_id)

            history = validate_and_sort(
                user_id,
                history,
                args.order_field,
            )

            items = [
                interaction["asin"]
                for interaction in history
            ]

            train_items = items[:-2]

            # Only real training items enter the output catalogue.
            training_items.update(train_items)

            eligible = len(items) <= args.threshold

            write_jsonl(
                users_file,
                {
                    "user_id": user_id,
                    "original_length": len(items),
                    "eligible": eligible,
                    "train_items": train_items,
                },
            )

            # Reserve the final REAL TRAIN item for internal development.
            # Example: train [A, B, C, D] gives:
            # fitting: [A] -> B, [A, B] -> C
            # internal development: [A, B, C] -> D
            positions = list(
                range(1, len(train_items) - 1)
            )[-args.max_examples_per_user:]

            for position in positions:
                full_prefix = train_items[:position]

                write_jsonl(
                    train_file,
                    {
                        "user_id": user_id,
                        "history": full_prefix[-args.max_history:],
                        "target": train_items[position],
                    },
                )

                report["training_examples"] += 1
                report["repeated_training_targets"] += int(
                    train_items[position] in full_prefix
                )

            write_jsonl(
                dev_file,
                {
                    "user_id": user_id,
                    "history": train_items[:-1][-args.max_history:],
                    "target": train_items[-1],
                },
            )

            report["repeated_dev_targets"] += int(
                train_items[-1] in train_items[:-1]
            )

            report["users"] += 1
            report["eligible_users"] += int(eligible)
            report["original_interactions"] += len(items)
            report["real_training_interactions"] += len(train_items)
            report["internal_dev_examples"] += 1

            if report["users"] % 10_000 == 0:
                print(
                    f"Prepared {report['users']:,} users",
                    flush=True,
                )

    if report["users"] == 0 or report["eligible_users"] == 0:
        raise ValueError("No users or no eligible users found.")

    catalogue = sorted(training_items)

    (output_dir / "item_catalogue.json").write_text(
        json.dumps(catalogue, ensure_ascii=False),
        encoding="utf-8",
    )

    report["train_catalogue_items"] = len(catalogue)
    report["expected_synthetic_interactions"] = (
        report["eligible_users"] * 3
    )
    report["source_sha256"] = file_sha256(input_file)
    report["status"] = "PASS"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nOutput folder: {output_dir}")


if __name__ == "__main__":
    main()