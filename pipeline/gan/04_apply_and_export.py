"""
Apply selective GAN augmentation and export RecBole data.

Default sequence:
    real training -> GAN items -> original validation -> original test

Only eligible users receive three synthetic interactions.

RecBole's timestamp column contains sequence positions,
not the original wall-clock timestamps.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import ijson


DATASETS = {
    "fashion": ("Fashion", "Fashion"),
    "video_games": ("Video Games", "Video_Games"),
    "movies_tv": ("Movies and TV Shows", "Movies_TV"),
}


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def validate_history(user_id, history, order_field):
    if not isinstance(history, list) or len(history) < 5:
        raise ValueError(
            f"{user_id}: invalid original history."
        )

    for interaction in history:
        if not isinstance(interaction, dict):
            raise ValueError(
                f"{user_id}: invalid interaction."
            )

        asin = interaction.get("asin")

        if not isinstance(asin, str) or not asin.strip():
            raise ValueError(
                f"{user_id}: invalid asin."
            )

        source = str(
            interaction.get("interaction_source", "")
        ).lower()

        if "synthetic" in source:
            raise ValueError(
                f"{user_id}: input is already augmented."
            )

        order = interaction.get(order_field)

        if (
            isinstance(order, bool)
            or not isinstance(order, (int, float))
            or not math.isfinite(order)
        ):
            raise ValueError(
                f"{user_id}: invalid {order_field}."
            )

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
                f"{user_id}: original split markers conflict."
            )

    return history


def check_token(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Empty or invalid RecBole token.")

    if any(character in value for character in "\t\r\n"):
        raise ValueError(
            "RecBole tokens cannot contain tabs or newlines."
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
    )

    parser.add_argument(
        "--placement",
        choices=["before_holdouts", "prepend"],
        default="before_holdouts",
    )

    args = parser.parse_args()

    folder, prefix = DATASETS[args.dataset]

    project_dir = Path(__file__).resolve().parents[2]

    data_dir = (
        project_dir
        / "datasets"
        / folder
        / "processed"
        / "gan"
    )

    preparation_path = (
        data_dir / "preparation_report.json"
    )

    generation_path = (
        data_dir
        / "generated"
        / "generation_report.json"
    )

    synthetic_path = (
        data_dir
        / "generated"
        / "selective_items.jsonl"
    )

    preparation = json.loads(
        preparation_path.read_text()
    )

    generation = json.loads(
        generation_path.read_text()
    )

    if (
        preparation["status"] != "PASS"
        or generation["status"] != "PASS"
        or preparation["dataset"] != folder
        or generation["dataset"] != folder
    ):
        raise ValueError(
            "Invalid preparation/generation reports."
        )

    if (
        generation["threshold"] != preparation["threshold"]
        or generation["synthetic_per_user"] != 3
        or generation["eligible_users"]
        != preparation["eligible_users"]
    ):
        raise ValueError(
            "Generation settings/counts disagree "
            "with preparation."
        )

    input_path = Path(preparation["input_file"])

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    original_hash = sha256(input_path)

    if (
        original_hash != preparation["source_sha256"]
        or original_hash != generation["source_sha256"]
    ):
        raise ValueError(
            "Original histories changed since preparation."
        )

    if (
        sha256(synthetic_path)
        != generation["output_sha256"]
    ):
        raise ValueError(
            "Generated items changed since generation."
        )

    catalogue = set(
        json.loads(
            (data_dir / "item_catalogue.json").read_text()
        )
    )

    generated = {}

    with synthetic_path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            user_id = row["user_id"]
            items = row["items"]

            if user_id in generated:
                raise ValueError(
                    f"Duplicate generated user: {user_id}"
                )

            if (
                not isinstance(items, list)
                or len(items) != 3
                or any(
                    not isinstance(item, str)
                    for item in items
                )
                or len(set(items)) != 3
            ):
                raise ValueError(
                    f"{user_id}: expected three distinct items."
                )

            if set(items) - catalogue:
                raise ValueError(
                    f"{user_id}: generated item outside catalogue."
                )

            generated[user_id] = items

    if len(generated) != preparation["eligible_users"]:
        raise ValueError(
            "Generated user count mismatch."
        )

    dataset_name = (
        f"{prefix}_selective_gan_augmented"
    )

    recbole_dir = (
        project_dir
        / "datasets"
        / dataset_name
    )

    inter_path = (
        recbole_dir / f"{dataset_name}.inter"
    )

    histories_path = (
        project_dir
        / "datasets"
        / folder
        / "exports"
        / f"{dataset_name}_user_histories.json"
    )

    report_path = data_dir / "export_report.json"

    inter_temp = inter_path.with_suffix(
        ".inter.partial"
    )

    histories_temp = histories_path.with_suffix(
        ".json.partial"
    )

    for path in (
        inter_path,
        histories_path,
        report_path,
        inter_temp,
        histories_temp,
    ):
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )

    recbole_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    histories_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    counts = Counter()
    seen_users = set()
    holdout_overlap_examples = []

    try:
        with (
            input_path.open("rb") as source_file,
            inter_temp.open(
                "w", encoding="utf-8"
            ) as inter_file,
            histories_temp.open(
                "w", encoding="utf-8"
            ) as histories_file,
        ):
            inter_file.write(
                "user_id:token\t"
                "item_id:token\t"
                "timestamp:float\n"
            )

            histories_file.write("{\n")

            source_users = ijson.kvitems(
                source_file,
                "",
                use_float=True,
            )

            for user_id, original in source_users:
                check_token(user_id)

                if user_id in seen_users:
                    raise ValueError(
                        f"Duplicate original user: {user_id}"
                    )

                seen_users.add(user_id)

                original = validate_history(
                    user_id,
                    original,
                    preparation["order_field"],
                )

                eligible = (
                    len(original)
                    <= preparation["threshold"]
                )

                additions = generated.pop(
                    user_id,
                    None,
                )

                if eligible and additions is None:
                    raise ValueError(
                        f"{user_id}: missing generated items."
                    )

                if not eligible and additions is not None:
                    raise ValueError(
                        f"{user_id}: ineligible user was augmented."
                    )

                additions = additions or []

                training_items = {
                    interaction["asin"]
                    for interaction in original[:-2]
                }

                if set(additions) & training_items:
                    raise ValueError(
                        f"{user_id}: synthetic item "
                        "overlaps real training history."
                    )

                # Audit holdout collisions without using
                # holdout identities to filter generation.
                validation_overlap = (
                    original[-2]["asin"] in additions
                )

                test_overlap = (
                    original[-1]["asin"] in additions
                )

                counts["synthetic_validation_overlap_users"] += int(
                    validation_overlap
                )

                counts["synthetic_test_overlap_users"] += int(
                    test_overlap
                )

                if (
                    validation_overlap or test_overlap
                ) and len(holdout_overlap_examples) < 20:
                    holdout_overlap_examples.append(
                        {
                            "user_id": user_id,
                            "validation_overlap": validation_overlap,
                            "test_overlap": test_overlap,
                        }
                    )

                real_records = []

                for index, interaction in enumerate(original):
                    record = dict(interaction)

                    # Retain original ordering/source fields
                    # when replacing them for this export.
                    for field in (
                        "sequence_position",
                        "interaction_source",
                    ):
                        if field in record:
                            backup = f"original_{field}"

                            if backup in record:
                                raise ValueError(
                                    f"{user_id}: reserved "
                                    f"backup field already exists: {backup}"
                                )

                            record[backup] = record[field]

                    if index < len(original) - 2:
                        source = "real_train"
                    elif index == len(original) - 2:
                        source = "real_validation"
                    else:
                        source = "real_test"

                    record["interaction_source"] = source
                    real_records.append(record)

                synthetic_records = [
                    {
                        "asin": item,
                        "interaction_source": "gan_synthetic",
                        "augmentation_method": "history_conditional_gan",
                        "synthetic_position": index + 1,
                    }
                    for index, item in enumerate(additions)
                ]

                if args.placement == "prepend":
                    augmented = (
                        synthetic_records + real_records
                    )
                else:
                    augmented = (
                        real_records[:-2]
                        + synthetic_records
                        + real_records[-2:]
                    )

                retained_real = [
                    record
                    for record in augmented
                    if record["interaction_source"]
                    != "gan_synthetic"
                ]

                if [
                    record["asin"]
                    for record in retained_real
                ] != [
                    record["asin"]
                    for record in original
                ]:
                    raise ValueError(
                        f"{user_id}: original item order changed."
                    )

                # Verify original payload fields were retained.
                for original_record, exported_record in zip(
                    original,
                    retained_real,
                ):
                    for field, value in original_record.items():
                        exported_field = (
                            f"original_{field}"
                            if field in (
                                "sequence_position",
                                "interaction_source",
                            )
                            else field
                        )

                        if exported_record.get(exported_field) != value:
                            raise ValueError(
                                f"{user_id}: original field "
                                f"changed: {field}"
                            )

                if [
                    record["interaction_source"]
                    for record in augmented[-2:]
                ] != ["real_validation", "real_test"]:
                    raise ValueError(
                        f"{user_id}: holdout boundaries corrupted."
                    )

                for position, record in enumerate(
                    augmented,
                    start=1,
                ):
                    check_token(record["asin"])

                    record["sequence_position"] = position

                    inter_file.write(
                        f"{user_id}\t"
                        f"{record['asin']}\t"
                        f"{float(position)}\n"
                    )

                if counts["users"]:
                    histories_file.write(",\n")

                histories_file.write(
                    json.dumps(user_id)
                    + ": "
                    + json.dumps(
                        augmented,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )

                counts["users"] += 1
                counts["eligible_users"] += int(eligible)
                counts["original_interactions"] += len(original)
                counts["synthetic_interactions"] += len(additions)
                counts["final_interactions"] += len(augmented)

            histories_file.write("\n}\n")

        if generated:
            raise ValueError(
                "Generated output contains unexpected users."
            )

        for field in (
            "users",
            "eligible_users",
            "original_interactions",
        ):
            if counts[field] != preparation[field]:
                raise ValueError(
                    f"Count mismatch: {field}"
                )

        if (
            counts["synthetic_interactions"]
            != preparation["expected_synthetic_interactions"]
        ):
            raise ValueError(
                "Synthetic interaction count mismatch."
            )

        if (
            counts["final_interactions"]
            != counts["original_interactions"]
            + counts["synthetic_interactions"]
        ):
            raise ValueError(
                "Final interaction count mismatch."
            )

        inter_temp.replace(inter_path)
        histories_temp.replace(histories_path)

    except BaseException:
        inter_temp.unlink(missing_ok=True)
        histories_temp.unlink(missing_ok=True)
        raise

    report = {
        "status": "PASS",
        "dataset": dataset_name,
        "placement": args.placement,
        "threshold": preparation["threshold"],
        **counts,
        "synthetic_validation_overlap_users": counts[
            "synthetic_validation_overlap_users"
        ],
        "synthetic_test_overlap_users": counts[
            "synthetic_test_overlap_users"
        ],
        "holdout_overlap_examples": holdout_overlap_examples,
        "inter_file": str(inter_path),
        "histories_file": str(histories_path),
        "source_sha256": original_hash,
        "inter_sha256": sha256(inter_path),
        "holdout_overlap_policy": (
            "Audit only; no filtering using heldout identities."
        ),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("\nSelective GAN export complete.")


if __name__ == "__main__":
    main()