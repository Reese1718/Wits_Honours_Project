"""
Generate selective augmentation using the trained GAN.

Input:
    processed/gan/users.jsonl
    processed/gan/checkpoints/best_generator.pt

Output:
    processed/gan/generated/selective_items.jsonl
    processed/gan/generated/generation_report.json

Only users with original history length <= the preparation
threshold receive synthetic items.

Official validation/test item identities are not used.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from gan_models import Generator


FOLDERS = {
    "fashion": "Fashion",
    "video_games": "Video Games",
    "movies_tv": "Movies and TV Shows",
}

SYNTHETIC_COUNT = 3


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=FOLDERS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )

    args = parser.parse_args()

    if (
        args.batch_size < 1
        or args.candidates < SYNTHETIC_COUNT
        or not np.isfinite(args.temperature)
        or args.temperature <= 0
    ):
        parser.error(
            "Use a positive batch size and temperature, "
            "and at least three candidates."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Select a Colab GPU runtime."
        )

    project_dir = Path(__file__).resolve().parents[2]

    data_dir = (
        project_dir
        / "datasets"
        / FOLDERS[args.dataset]
        / "processed"
        / "gan"
    )

    checkpoint_path = (
        data_dir
        / "checkpoints"
        / "best_generator.pt"
    )

    users_path = data_dir / "users.jsonl"
    output_dir = data_dir / "generated"

    items_path = output_dir / "selective_items.jsonl"
    report_path = output_dir / "generation_report.json"
    temporary_path = output_dir / "selective_items.jsonl.partial"

    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (
        items_path,
        report_path,
        temporary_path,
    ):
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    preparation = json.loads(
        (data_dir / "preparation_report.json").read_text()
    )

    if (
        preparation["status"] != "PASS"
        or preparation["dataset"] != FOLDERS[args.dataset]
    ):
        raise ValueError(
            "Invalid preparation report."
        )

    # Load only your own locally generated checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint["settings"]["dataset"] != args.dataset:
        raise ValueError(
            "Checkpoint belongs to another dataset."
        )

    if checkpoint["metrics"]["phase"] != "GAN":
        raise ValueError(
            "Use a checkpoint from adversarial training."
        )

    # Ensure generation inputs match those used during training.
    for name in (
        "users.jsonl",
        "item_catalogue.json",
        "preparation_report.json",
    ):
        if (
            sha256(data_dir / name)
            != checkpoint["fingerprints"][name]
        ):
            raise ValueError(
                f"{name} changed after training."
            )

    catalogue = checkpoint["catalogue"]

    mapping = {
        item: index + 1
        for index, item in enumerate(catalogue)
    }

    if len(mapping) != len(catalogue):
        raise ValueError("Duplicate catalogue items.")

    all_ids = np.arange(
        1,
        len(catalogue) + 1,
        dtype=np.int64,
    )

    settings = checkpoint["settings"]
    max_history = checkpoint["max_history"]
    device = torch.device(args.device)

    generator = Generator(
        len(catalogue) + 1,
        settings["embedding_size"],
        settings["hidden_size"],
    )

    generator.load_state_dict(
        checkpoint["generator"]
    )

    generator.to(device)
    generator.eval()

    rng = np.random.default_rng(args.seed)

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    counts = Counter()
    seen_users = set()
    users_processed = 0
    generated_users = 0
    pending = []
    start_time = time.perf_counter()

    @torch.no_grad()
    def generate_batch(rows, output_file):
        nonlocal generated_users

        histories = []
        candidate_pools = []

        for row in rows:
            full_history = [
                mapping[item]
                for item in row["train_items"]
            ]

            histories.append(
                full_history[-max_history:]
            )

            # Exclude ALL real training items, including
            # those beyond the model's history window.
            seen = set(full_history)

            available_count = len(catalogue) - len(seen)
            candidate_count = min(
                args.candidates,
                available_count,
            )

            if candidate_count < SYNTHETIC_COUNT:
                raise ValueError(
                    f"{row['user_id']}: fewer than three "
                    "unseen training-catalogue items."
                )

            selected = set()

            for _ in range(100):
                draws = rng.choice(
                    all_ids,
                    size=2 * candidate_count,
                    replace=True,
                )

                selected.update(
                    int(item)
                    for item in draws
                    if int(item) not in seen
                )

                if len(selected) >= candidate_count:
                    break

            if len(selected) < candidate_count:
                selected.update(
                    int(item)
                    for item in all_ids
                    if int(item) not in seen
                )

            pool = np.asarray(
                sorted(selected),
                dtype=np.int64,
            )

            candidate_pools.append(
                rng.choice(
                    pool,
                    size=candidate_count,
                    replace=False,
                )
            )

        lengths = np.asarray(
            [len(history) for history in histories],
            dtype=np.int64,
        )

        if (lengths == 0).any():
            raise ValueError("Empty training history.")

        padded_history = np.zeros(
            (len(rows), int(lengths.max())),
            dtype=np.int64,
        )

        for index, history in enumerate(histories):
            padded_history[index, :len(history)] = history

        width = max(
            len(pool)
            for pool in candidate_pools
        )

        padded_candidates = np.zeros(
            (len(rows), width),
            dtype=np.int64,
        )

        valid_candidates = np.zeros(
            (len(rows), width),
            dtype=bool,
        )

        for index, pool in enumerate(candidate_pools):
            padded_candidates[index, :len(pool)] = pool
            valid_candidates[index, :len(pool)] = True

        history_tensor = torch.as_tensor(
            padded_history,
            device=device,
        )
        length_tensor = torch.as_tensor(
            lengths,
            device=device,
        )
        candidate_tensor = torch.as_tensor(
            padded_candidates,
            device=device,
        )
        mask_tensor = torch.as_tensor(
            valid_candidates,
            device=device,
        )

        logits = generator(
            history_tensor,
            length_tensor,
            candidate_tensor,
        )

        logits = logits / args.temperature

        if not torch.isfinite(logits[mask_tensor]).all():
            raise RuntimeError(
                "Non-finite generation scores."
            )

        logits = logits.masked_fill(
            ~mask_tensor,
            -torch.inf,
        )

        # Gumbel top-k samples without replacement,
        # producing three distinct candidate IDs.
        uniform = torch.rand_like(logits).clamp(
            min=1e-6,
            max=1.0 - 1e-6,
        )

        noise = -torch.log(
            -torch.log(uniform)
        )

        positions = torch.topk(
            logits + noise,
            k=SYNTHETIC_COUNT,
            dim=-1,
        ).indices

        chosen_ids = candidate_tensor.gather(
            1,
            positions,
        ).cpu().tolist()

        for row, ids in zip(rows, chosen_ids):
            items = [
                catalogue[item_id - 1]
                for item_id in ids
            ]

            if (
                len(set(items)) != SYNTHETIC_COUNT
                or set(items) & set(row["train_items"])
            ):
                raise ValueError(
                    f"{row['user_id']}: duplicates "
                    "or training overlap."
                )

            output_file.write(
                json.dumps(
                    {
                        "user_id": row["user_id"],
                        "items": items,
                        "method": "history_conditional_gan",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            counts.update(items)
            generated_users += 1

        print(
            f"Generated {generated_users:,}/"
            f"{preparation['eligible_users']:,} users",
            flush=True,
        )

    try:
        with (
            users_path.open(encoding="utf-8") as source,
            temporary_path.open(
                "w",
                encoding="utf-8",
            ) as output_file,
        ):
            for line in source:
                row = json.loads(line)
                user_id = row["user_id"]

                if user_id in seen_users:
                    raise ValueError(
                        f"Duplicate user: {user_id}"
                    )

                seen_users.add(user_id)
                users_processed += 1

                if (
                    len(row["train_items"])
                    != row["original_length"] - 2
                ):
                    raise ValueError(
                        f"{user_id}: training length mismatch."
                    )

                expected_eligibility = (
                    row["original_length"]
                    <= preparation["threshold"]
                )

                if row["eligible"] != expected_eligibility:
                    raise ValueError(
                        f"{user_id}: eligibility mismatch."
                    )

                if not row["eligible"]:
                    continue

                pending.append(row)

                if len(pending) == args.batch_size:
                    generate_batch(
                        pending,
                        output_file,
                    )
                    pending = []

            if pending:
                generate_batch(
                    pending,
                    output_file,
                )

        if users_processed != preparation["users"]:
            raise ValueError("Total user count mismatch.")

        if generated_users != preparation["eligible_users"]:
            raise ValueError("Eligible user count mismatch.")

        temporary_path.replace(items_path)

    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    report = {
        "status": "PASS",
        "dataset": FOLDERS[args.dataset],
        "users_processed": users_processed,
        "eligible_users": generated_users,
        "threshold": preparation["threshold"],
        "synthetic_per_user": SYNTHETIC_COUNT,
        "synthetic_interactions": (
            generated_users * SYNTHETIC_COUNT
        ),
        "unique_generated_items": len(counts),
        "most_common_generated_items": counts.most_common(20),
        "checkpoint_epoch": checkpoint["epoch"] + 1,
        "checkpoint_sha256": sha256(checkpoint_path),
        "source_sha256": preparation["source_sha256"],
        "output_sha256": sha256(items_path),
        "seed": args.seed,
        "candidates": args.candidates,
        "temperature": args.temperature,
        "seconds": time.perf_counter() - start_time,
        "holdout_policy": (
            "Official validation/test identities "
            "were not used for generation or exclusion."
        ),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nGenerated items: {items_path}")


if __name__ == "__main__":
    main()