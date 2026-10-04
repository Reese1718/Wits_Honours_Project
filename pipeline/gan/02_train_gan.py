"""
Train a history-conditioned GAN inspired by AugCF.

Reads only the prepared GAN data.
Official validation/test records are not read.

Phase 1:
    Pretrain the generator on real next-item examples.

Phase 2:
    Alternate discriminator and generator updates.
    Retain supervised loss alongside adversarial loss.

Checkpoint selection:
    Lowest candidate cross-entropy on the internal development split.
    Only GAN-phase checkpoints qualify as best_generator.pt.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from gan_models import Generator, Discriminator


FOLDERS = {
    "fashion": "Fashion",
    "video_games": "Video Games",
    "movies_tv": "Movies and TV Shows",
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(path, mapping, max_history):
    histories = []
    targets = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)

            ids = [
                mapping[item]
                for item in row["history"]
            ]

            if not ids or len(ids) > max_history:
                raise ValueError(
                    f"Invalid history in {path}"
                )

            histories.append(ids)
            targets.append(mapping[row["target"]])

    if not histories:
        raise ValueError(f"No examples in {path}")

    padded = np.zeros(
        (len(histories), max_history),
        dtype=np.int64,
    )

    lengths = np.asarray(
        [len(history) for history in histories],
        dtype=np.int64,
    )

    for index, history in enumerate(histories):
        padded[index, :len(history)] = history

    return (
        padded,
        lengths,
        np.asarray(targets, dtype=np.int64),
    )


def make_batch(
    rows,
    indices,
    n_items,
    candidate_count,
    rng,
    device,
):
    histories, lengths, targets = (
        array[indices]
        for array in rows
    )

    candidates = np.empty(
        (len(indices), candidate_count),
        dtype=np.int64,
    )

    for index, (history, length, target) in enumerate(
        zip(histories, lengths, targets)
    ):
        # The positive target is always candidate 0.
        # Other candidates exclude items in this context
        # and exclude the positive target.
        banned = set(
            history[:length].tolist()
        ) | {int(target)}

        available_count = n_items - 1 - len(banned)

        if available_count < candidate_count - 1:
            raise ValueError(
                "Candidate count exceeds the available "
                "unseen items. Reduce --candidates."
            )

        negatives = set()

        while len(negatives) < candidate_count - 1:
            draws = rng.integers(
                1,
                n_items,
                size=2 * candidate_count,
            )

            negatives.update(
                int(item)
                for item in draws
                if int(item) not in banned
            )

        pool = np.asarray(
            sorted(negatives),
            dtype=np.int64,
        )

        candidates[index, 0] = target

        candidates[index, 1:] = rng.choice(
            pool,
            size=candidate_count - 1,
            replace=False,
        )

    return (
        torch.as_tensor(histories, device=device),
        torch.as_tensor(lengths, device=device),
        torch.as_tensor(candidates, device=device),
    )


@torch.no_grad()
def evaluate(
    generator,
    rows,
    indices,
    n_items,
    args,
    device,
):
    generator.eval()

    # Fixed candidates make epochs comparable.
    rng = np.random.default_rng(
        args.seed + 100000
    )

    total_loss = 0.0
    hits = 0
    count = 0

    for start in range(
        0,
        len(indices),
        args.batch_size,
    ):
        batch_indices = indices[
            start:start + args.batch_size
        ]

        histories, lengths, candidates = make_batch(
            rows,
            batch_indices,
            n_items,
            args.candidates,
            rng,
            device,
        )

        logits = generator(
            histories,
            lengths,
            candidates,
        )

        target = torch.zeros(
            len(batch_indices),
            dtype=torch.long,
            device=device,
        )

        total_loss += F.cross_entropy(
            logits,
            target,
            reduction="sum",
        ).item()

        hits += (
            logits.argmax(-1) == 0
        ).sum().item()

        count += len(batch_indices)

    return {
        "dev_candidate_ce": total_loss / count,
        "dev_candidate_hit1": hits / count,
        "dev_examples": count,
    }


def update(loss, model, optimizer):
    if not torch.isfinite(loss):
        raise RuntimeError(
            "Non-finite loss; training stopped."
        )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
    )

    optimizer.step()


def save_checkpoint(path, state):
    temporary = path.with_suffix(".pt.partial")

    torch.save(state, temporary)

    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=FOLDERS,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--gan-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    parser.add_argument("--max-dev-examples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
    )
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if (
        min(
            args.batch_size,
            args.embedding_size,
            args.hidden_size,
            args.max_dev_examples,
        ) < 1
        or args.candidates < 2
    ):
        parser.error(
            "Positive sizes and at least two candidates are required."
        )

    if (
        args.pretrain_epochs < 1
        or args.gan_epochs < 1
        or args.learning_rate <= 0
        or args.adversarial_weight <= 0
    ):
        parser.error(
            "Epochs, learning rate and adversarial weight "
            "must be positive."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Select a Colab GPU runtime before training."
        )

    device = torch.device(args.device)

    project_dir = Path(__file__).resolve().parents[2]

    data_dir = (
        project_dir
        / "datasets"
        / FOLDERS[args.dataset]
        / "processed"
        / "gan"
    )

    report = json.loads(
        (data_dir / "preparation_report.json").read_text()
    )

    if (
        report["status"] != "PASS"
        or report["dataset"] != FOLDERS[args.dataset]
    ):
        raise ValueError(
            "Preparation report is incomplete "
            "or belongs to another dataset."
        )

    catalogue = json.loads(
        (data_dir / "item_catalogue.json").read_text()
    )

    if (
        len(catalogue) != len(set(catalogue))
        or len(catalogue) != report["train_catalogue_items"]
    ):
        raise ValueError("Invalid item catalogue.")

    mapping = {
        item: index + 1
        for index, item in enumerate(catalogue)
    }

    n_items = len(mapping) + 1

    train_rows = load_examples(
        data_dir / "train_examples.jsonl",
        mapping,
        report["max_history"],
    )

    dev_rows = load_examples(
        data_dir / "dev_examples.jsonl",
        mapping,
        report["max_history"],
    )

    if (
        len(train_rows[0]) != report["training_examples"]
        or len(dev_rows[0]) != report["internal_dev_examples"]
    ):
        raise ValueError(
            "Prepared example counts differ from the report."
        )

    fingerprints = {
        name: sha256(data_dir / name)
        for name in [
            "preparation_report.json",
            "item_catalogue.json",
            "train_examples.jsonl",
            "dev_examples.jsonl",
            "users.jsonl",
        ]
    }

    dev_indices = np.random.default_rng(
        args.seed
    ).choice(
        len(dev_rows[0]),
        size=min(
            args.max_dev_examples,
            len(dev_rows[0]),
        ),
        replace=False,
    )

    checkpoint_dir = data_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    if not args.resume and any(checkpoint_dir.iterdir()):
        raise FileExistsError(
            "Checkpoints already exist. "
            "Use --resume to continue."
        )

    settings = {
        key: value
        for key, value in vars(args).items()
        if key not in ("resume", "device")
    }

    set_seed(args.seed)

    generator = Generator(
        n_items,
        args.embedding_size,
        args.hidden_size,
    ).to(device)

    discriminator = Discriminator(
        n_items,
        args.embedding_size,
        args.hidden_size,
    ).to(device)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.learning_rate,
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate,
    )

    start_epoch = 0
    best_ce = float("inf")

    if args.resume:
        state = torch.load(
            checkpoint_dir / "last.pt",
            map_location=device,
            weights_only=False,
        )

        if (
            state["settings"] != settings
            or state["fingerprints"] != fingerprints
        ):
            raise ValueError(
                "Resume settings or prepared data have changed."
            )

        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])

        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])

        start_epoch = state["epoch"] + 1
        best_ce = state["best_ce"]

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Items: {len(catalogue):,}; "
        f"training examples: {len(train_rows[0]):,}"
    )

    total_epochs = (
        args.pretrain_epochs + args.gan_epochs
    )

    for epoch in range(start_epoch, total_epochs):
        set_seed(args.seed + epoch)

        rng = np.random.default_rng(
            args.seed + epoch
        )

        order = rng.permutation(
            len(train_rows[0])
        )

        adversarial = epoch >= args.pretrain_epochs
        phase = "GAN" if adversarial else "PRETRAIN"

        gan_epoch = max(
            0,
            epoch - args.pretrain_epochs,
        )

        # Anneal Gumbel temperature from 0.9 to 0.5.
        temperature = (
            0.9
            - 0.4 * gan_epoch
            / max(1, args.gan_epochs - 1)
        )

        generator.train()
        discriminator.train()

        sums = np.zeros(4)
        example_count = 0
        start_time = time.perf_counter()

        batches = tqdm(
            range(0, len(order), args.batch_size),
            desc=f"{phase} epoch {epoch + 1}/{total_epochs}",
        )

        for start in batches:
            indices = order[
                start:start + args.batch_size
            ]

            histories, lengths, candidates = make_batch(
                train_rows,
                indices,
                n_items,
                args.candidates,
                rng,
                device,
            )

            target = torch.zeros(
                len(indices),
                dtype=torch.long,
                device=device,
            )

            discriminator_value = 0.0
            adversarial_value = 0.0

            if adversarial:
                # Update discriminator.
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)

                optimizer_d.zero_grad(set_to_none=True)

                with torch.no_grad():
                    sample = F.gumbel_softmax(
                        generator(
                            histories,
                            lengths,
                            candidates,
                        ),
                        tau=temperature,
                        hard=True,
                    )

                item_embeddings = (
                    discriminator.item_embedding(candidates)
                )

                fake_embedding = (
                    sample.unsqueeze(-1) * item_embeddings
                ).sum(1)

                real_score = discriminator(
                    histories,
                    lengths,
                    item_embeddings[:, 0],
                )

                fake_score = discriminator(
                    histories,
                    lengths,
                    fake_embedding,
                )

                discriminator_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        real_score,
                        torch.ones_like(real_score),
                    )
                    + F.binary_cross_entropy_with_logits(
                        fake_score,
                        torch.zeros_like(fake_score),
                    )
                )

                update(
                    discriminator_loss,
                    discriminator,
                    optimizer_d,
                )

                discriminator_value = (
                    discriminator_loss.item()
                )

                # Freeze its parameters while allowing gradients
                # through its input to reach the generator.
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)

            # Update generator.
            optimizer_g.zero_grad(set_to_none=True)

            logits = generator(
                histories,
                lengths,
                candidates,
            )

            supervised_loss = F.cross_entropy(
                logits,
                target,
            )

            generator_loss = supervised_loss

            if adversarial:
                sample = F.gumbel_softmax(
                    logits,
                    tau=temperature,
                    hard=True,
                )

                fake_embedding = (
                    sample.unsqueeze(-1)
                    * discriminator.item_embedding(candidates)
                ).sum(1)

                fake_score = discriminator(
                    histories,
                    lengths,
                    fake_embedding,
                )

                adversarial_loss = (
                    F.binary_cross_entropy_with_logits(
                        fake_score,
                        torch.ones_like(fake_score),
                    )
                )

                generator_loss = (
                    supervised_loss
                    + args.adversarial_weight * adversarial_loss
                )

                adversarial_value = adversarial_loss.item()

            update(
                generator_loss,
                generator,
                optimizer_g,
            )

            sums += len(indices) * np.asarray(
                [
                    generator_loss.item(),
                    supervised_loss.item(),
                    discriminator_value,
                    adversarial_value,
                ]
            )

            example_count += len(indices)

        metrics = evaluate(
            generator,
            dev_rows,
            dev_indices,
            n_items,
            args,
            device,
        )

        improved = (
            adversarial
            and metrics["dev_candidate_ce"] < best_ce
        )

        if improved:
            best_ce = metrics["dev_candidate_ce"]

        record = {
            "epoch": epoch + 1,
            "phase": phase,
            "seconds": time.perf_counter() - start_time,
            "generator_loss": sums[0] / example_count,
            "supervised_ce": sums[1] / example_count,
            "discriminator_loss": sums[2] / example_count,
            "generator_adversarial_loss": sums[3] / example_count,
            "gumbel_temperature": temperature,
            **metrics,
        }

        print(
            json.dumps(record, indent=2),
            flush=True,
        )

        with (
            checkpoint_dir / "training_log.jsonl"
        ).open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        state = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "best_ce": best_ce,
            "settings": settings,
            "fingerprints": fingerprints,
            "catalogue": catalogue,
            "max_history": report["max_history"],
            "metrics": record,
        }

        if improved:
            save_checkpoint(
                checkpoint_dir / "best_generator.pt",
                state,
            )

        if epoch == args.pretrain_epochs - 1:
            save_checkpoint(
                checkpoint_dir / "pretrained_generator.pt",
                state,
            )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            state,
        )

    print(
        "\nFinished. Best GAN checkpoint:\n"
        f"{checkpoint_dir / 'best_generator.pt'}"
    )


if __name__ == "__main__":
    main()