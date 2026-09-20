"""
Select the best QLoRA checkpoint using validation NDCG@10.

The script:

1. Loads every checkpoint produced by Step 04.
2. Builds validation prompts using real chronological histories.
3. Uses full-catalogue constrained beam search.
4. Excludes items already present in each user's history.
5. Calculates validation ranking metrics.
6. Selects the checkpoint with the highest NDCG@10.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from common import (
    build_recommendation_messages,
    iter_jsonl,
    load_metadata_lookup,
    load_yaml,
    read_json,
    resolve_project_path,
    set_random_seed,
    write_json,
    write_jsonl,
)
from constrained_decoding import CatalogueTokenConstraint


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best QLoRA checkpoint using NDCG@10."
    )

    parser.add_argument(
        "config",
        help="QLoRA experiment YAML file used for Step 04.",
    )

    return parser.parse_args()


def nested_value(
    config: dict[str, Any],
    path: str,
    default: Any = None,
    required: bool = False,
) -> Any:
    current: Any = config

    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            if required:
                raise KeyError(
                    f"Missing required configuration value: {path}"
                )

            return default

        current = current[component]

    return current


def select_compute_dtype(value: str) -> torch.dtype:
    requested = value.lower()

    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    if requested in {"bfloat16", "bf16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "The selected GPU does not support bfloat16."
            )

        return torch.bfloat16

    if requested in {"float16", "fp16"}:
        return torch.float16

    raise ValueError(
        "compute_dtype must be auto, bfloat16, or float16."
    )


def checkpoint_number(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)

    if match is None:
        return -1

    return int(match.group(1))


def find_checkpoints(output_directory: Path) -> list[Path]:
    checkpoints = [
        path
        for path in output_directory.glob("checkpoint-*")
        if path.is_dir()
        and (path / "adapter_config.json").exists()
    ]

    checkpoints.sort(key=checkpoint_number)

    if not checkpoints:
        fallback = (
            output_directory / "best_validation_loss_adapter"
        )

        if (
            fallback.exists()
            and (fallback / "adapter_config.json").exists()
        ):
            return [fallback]

        raise FileNotFoundError(
            f"No QLoRA checkpoints found under {output_directory}."
        )

    return checkpoints


def load_catalogue_item_ids(path: Path) -> list[str]:
    catalogue = read_json(path)
    items = catalogue.get("items")

    if not isinstance(items, list):
        raise ValueError(
            "Catalogue must contain an 'items' list."
        )

    item_ids = [
        str(item["item_id"])
        for item in items
        if isinstance(item, dict) and item.get("item_id")
    ]

    declared_count = catalogue.get("num_items")

    if (
        declared_count is not None
        and declared_count != len(item_ids)
    ):
        raise ValueError(
            "Catalogue item count does not match num_items."
        )

    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Catalogue contains duplicate item IDs.")

    return item_ids


def load_validation_records(
    path: Path,
    max_users: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    records = list(iter_jsonl(path))

    if not records:
        raise ValueError("Validation split is empty.")

    required_fields = {
        "user_id",
        "history_item_ids",
        "target_item_id",
    }

    for record_index, record in enumerate(records, start=1):
        missing_fields = required_fields.difference(record)

        if missing_fields:
            raise ValueError(
                f"Validation record {record_index} is missing: "
                f"{sorted(missing_fields)}"
            )

        history = record["history_item_ids"]

        if not isinstance(history, list) or not history:
            raise ValueError(
                f"Validation record {record_index} must contain a "
                "non-empty history_item_ids list."
            )

        if record["target_item_id"] is None:
            raise ValueError(
                f"Validation record {record_index} has no target_item_id."
            )

    if max_users is not None and len(records) > max_users:
        generator = random.Random(seed)
        selected_indices = sorted(
            generator.sample(
                range(len(records)),
                max_users,
            )
        )

        records = [
            records[index]
            for index in selected_indices
        ]

    return records


class HistoryAwareConstraint:
    """
    Full-catalogue constraint that removes each user's history items.

    It avoids rebuilding a 12k/54k-item trie for every user.
    """

    def __init__(
        self,
        catalogue_constraint: CatalogueTokenConstraint,
    ) -> None:
        self.constraint = catalogue_constraint
        self.global_prefix_counts: Counter[tuple[int, ...]] = Counter()

        for token_sequence in (
            self.constraint.item_to_token_sequence.values()
        ):
            for length in range(1, len(token_sequence) + 1):
                prefix = token_sequence[:length]
                self.global_prefix_counts[prefix] += 1

    def build_callback(
        self,
        prompt_length: int,
        blocked_items_by_batch: list[set[str]],
    ) -> Callable[[int, torch.Tensor], list[int]]:
        blocked_prefix_counts: list[
            Counter[tuple[int, ...]]
        ] = []

        blocked_exact_sequences: list[
            set[tuple[int, ...]]
        ] = []

        for blocked_items in blocked_items_by_batch:
            prefix_counts: Counter[tuple[int, ...]] = Counter()
            exact_sequences: set[tuple[int, ...]] = set()

            for item_id in blocked_items:
                token_sequence = (
                    self.constraint.item_to_token_sequence.get(
                        item_id
                    )
                )

                if token_sequence is None:
                    continue

                exact_sequences.add(token_sequence)

                for length in range(
                    1,
                    len(token_sequence) + 1,
                ):
                    prefix_counts[
                        token_sequence[:length]
                    ] += 1

            blocked_prefix_counts.append(prefix_counts)
            blocked_exact_sequences.append(exact_sequences)

        def prefix_allowed_tokens_fn(
            batch_id: int,
            input_ids: torch.Tensor,
        ) -> list[int]:
            generated_tokens = tuple(
                int(token_id)
                for token_id in input_ids[
                    prompt_length:
                ].tolist()
            )

            if (
                self.constraint.eos_token_id
                in generated_tokens
            ):
                eos_position = generated_tokens.index(
                    self.constraint.eos_token_id
                )
                generated_tokens = generated_tokens[
                    :eos_position
                ]

            node = self.constraint.trie.get_node(
                generated_tokens
            )

            if node is None:
                return [self.constraint.eos_token_id]

            allowed: list[int] = []
            blocked_counts = blocked_prefix_counts[batch_id]

            for next_token, child_node in node.children.items():
                del child_node

                child_prefix = (
                    generated_tokens + (next_token,)
                )

                globally_available = (
                    self.global_prefix_counts[child_prefix]
                )
                blocked_available = blocked_counts.get(
                    child_prefix,
                    0,
                )

                if globally_available > blocked_available:
                    allowed.append(next_token)

            is_complete_item = (
                generated_tokens
                in self.constraint.token_sequence_to_item
            )

            is_blocked_item = (
                generated_tokens
                in blocked_exact_sequences[batch_id]
            )

            if is_complete_item and not is_blocked_item:
                allowed.append(
                    self.constraint.eos_token_id
                )

            if not allowed:
                # This should occur only if every catalogue item has
                # been blocked for a user.
                return [self.constraint.eos_token_id]

            return allowed

        return prefix_allowed_tokens_fn


def extract_batch_rankings(
    generation_output,
    catalogue_constraint: CatalogueTokenConstraint,
    prompt_length: int,
    batch_size: int,
    num_return_sequences: int,
    top_k: int,
) -> list[list[dict[str, Any]]]:
    sequences = generation_output.sequences
    sequence_scores = generation_output.sequences_scores

    all_rankings: list[list[dict[str, Any]]] = []

    for batch_index in range(batch_size):
        start = batch_index * num_return_sequences
        end = start + num_return_sequences

        candidates: list[tuple[str, float]] = []

        for sequence_index in range(start, end):
            generated_tokens = sequences[
                sequence_index,
                prompt_length:,
            ].tolist()

            item_id = (
                catalogue_constraint.generated_tokens_to_item(
                    generated_tokens
                )
            )

            if item_id is None:
                continue

            score = float(
                sequence_scores[sequence_index].item()
            )

            candidates.append((item_id, score))

        candidates.sort(
            key=lambda candidate: candidate[1],
            reverse=True,
        )

        seen: set[str] = set()
        ranking: list[dict[str, Any]] = []

        for item_id, score in candidates:
            if item_id in seen:
                continue

            seen.add(item_id)

            ranking.append(
                {
                    "rank": len(ranking) + 1,
                    "item_id": item_id,
                    "score": score,
                }
            )

            if len(ranking) == top_k:
                break

        all_rankings.append(ranking)

    return all_rankings


def calculate_rank(
    ranked_items: list[dict[str, Any]],
    target_item_id: str,
    top_k: int,
) -> int | None:
    for recommendation in ranked_items[:top_k]:
        if recommendation["item_id"] == target_item_id:
            return int(recommendation["rank"])

    return None


def calculate_metrics(
    predictions: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    hit_sum = 0.0
    ndcg_sum = 0.0
    mrr_sum = 0.0
    short_ranking_count = 0

    for prediction in predictions:
        rank = prediction["target_rank"]

        if len(prediction["recommendations"]) < top_k:
            short_ranking_count += 1

        if rank is not None and rank <= top_k:
            hit_sum += 1.0
            ndcg_sum += 1.0 / math.log2(rank + 1)
            mrr_sum += 1.0 / rank

    user_count = len(predictions)

    return {
        "users": user_count,
        f"hit@{top_k}": hit_sum / user_count,
        f"recall@{top_k}": hit_sum / user_count,
        f"ndcg@{top_k}": ndcg_sum / user_count,
        f"mrr@{top_k}": mrr_sum / user_count,
        "short_ranking_users": short_ranking_count,
        "complete_ranking_rate": (
            (user_count - short_ranking_count) / user_count
        ),
    }


@torch.inference_mode()
def evaluate_adapter(
    model,
    tokenizer,
    validation_records: list[dict[str, Any]],
    metadata_lookup: dict[str, dict[str, Any]],
    catalogue_constraint: CatalogueTokenConstraint,
    history_constraint: HistoryAwareConstraint,
    max_history_length: int,
    max_sequence_length: int,
    batch_size: int,
    top_k: int,
    num_beams: int,
    length_penalty: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []

    max_item_tokens = max(
        len(tokens)
        for tokens in (
            catalogue_constraint
            .item_to_token_sequence
            .values()
        )
    )

    max_new_tokens = max_item_tokens + 1
    started_at = time.time()

    for batch_start in range(
        0,
        len(validation_records),
        batch_size,
    ):
        batch = validation_records[
            batch_start:batch_start + batch_size
        ]

        prompt_texts: list[str] = []
        blocked_items_by_batch: list[set[str]] = []

        for record in batch:
            history = [
                str(item_id)
                for item_id in record["history_item_ids"]
            ]

            model_history = history[-max_history_length:]

            messages = build_recommendation_messages(
                history_item_ids=model_history,
                metadata_lookup=metadata_lookup,
                max_history_length=max_history_length,
            )

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompt_texts.append(prompt_text)
            # The model sees only the most recent items, but ranking must
            # exclude the user's complete real history. Otherwise an older,
            # hidden history item could incorrectly be recommended again.
            blocked_items_by_batch.append(set(history))

        model_inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            add_special_tokens=False,
        )

        model_inputs = {
            name: tensor.to(model.device)
            for name, tensor in model_inputs.items()
        }

        prompt_length = model_inputs["input_ids"].shape[1]

        prefix_callback = history_constraint.build_callback(
            prompt_length=prompt_length,
            blocked_items_by_batch=blocked_items_by_batch,
        )

        generation_output = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=top_k,
            do_sample=False,
            early_stopping=True,
            length_penalty=length_penalty,
            eos_token_id=catalogue_constraint.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            prefix_allowed_tokens_fn=prefix_callback,
        )

        batch_rankings = extract_batch_rankings(
            generation_output=generation_output,
            catalogue_constraint=catalogue_constraint,
            prompt_length=prompt_length,
            batch_size=len(batch),
            num_return_sequences=top_k,
            top_k=top_k,
        )

        for record, ranking in zip(
            batch,
            batch_rankings,
        ):
            target_item_id = str(
                record["target_item_id"]
            )

            target_rank = calculate_rank(
                ranked_items=ranking,
                target_item_id=target_item_id,
                top_k=top_k,
            )

            predictions.append(
                {
                    "user_id": str(record["user_id"]),
                    "target_item_id": target_item_id,
                    "target_rank": target_rank,
                    "recommendations": ranking,
                }
            )

        completed = min(
            batch_start + len(batch),
            len(validation_records),
        )

        if completed % 100 == 0 or completed == len(
            validation_records
        ):
            print(
                f"  Evaluated {completed:,}/"
                f"{len(validation_records):,} users"
            )

    metrics = calculate_metrics(
        predictions=predictions,
        top_k=top_k,
    )

    metrics["elapsed_seconds"] = (
        time.time() - started_at
    )

    return metrics, predictions


def main() -> None:
    args = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Step 05 requires a CUDA GPU."
        )

    config_path = resolve_project_path(args.config)
    config = load_yaml(config_path)

    seed = int(
        nested_value(
            config,
            "experiment.seed",
            default=42,
        )
    )
    set_random_seed(seed)

    model_name = str(
        nested_value(
            config,
            "model.name",
            required=True,
        )
    )

    output_directory = resolve_project_path(
        nested_value(
            config,
            "training.output_dir",
            required=True,
        )
    )

    validation_path = resolve_project_path(
        nested_value(
            config,
            "data.validation_split_file",
            required=True,
        )
    )

    metadata_path = resolve_project_path(
        nested_value(
            config,
            "data.metadata_file",
            required=True,
        )
    )

    catalogue_path = resolve_project_path(
        nested_value(
            config,
            "data.catalogue_file",
            required=True,
        )
    )

    top_k = int(
        nested_value(
            config,
            "checkpoint_selection.top_k",
            default=10,
        )
    )

    num_beams = int(
        nested_value(
            config,
            "checkpoint_selection.num_beams",
            default=20,
        )
    )

    batch_size = int(
        nested_value(
            config,
            "checkpoint_selection.batch_size",
            default=4,
        )
    )

    max_validation_users = nested_value(
        config,
        "checkpoint_selection.max_validation_users",
        default=None,
    )

    if max_validation_users is not None:
        max_validation_users = int(
            max_validation_users
        )

    exclude_history = bool(
        nested_value(
            config,
            "checkpoint_selection.exclude_history_items",
            default=True,
        )
    )

    if not exclude_history:
        raise ValueError(
            "This pipeline requires exclusion of history items to "
            "match non-repeatable RecBole evaluation."
        )

    length_penalty = float(
        nested_value(
            config,
            "checkpoint_selection.length_penalty",
            default=1.0,
        )
    )

    max_sequence_length = int(
        nested_value(
            config,
            "training.max_sequence_length",
            default=2048,
        )
    )

    max_history_length = int(
        nested_value(
            config,
            "checkpoint_selection.max_history_length",
            default=5,
        )
    )

    if max_history_length < 1:
        raise ValueError(
            "checkpoint_selection.max_history_length must be at least 1."
        )

    if num_beams < top_k:
        raise ValueError(
            "num_beams must be at least as large as top_k."
        )

    checkpoints = find_checkpoints(output_directory)

    validation_records = load_validation_records(
        path=validation_path,
        max_users=max_validation_users,
        seed=seed,
    )

    metadata_lookup = load_metadata_lookup(metadata_path)
    catalogue_item_ids = load_catalogue_item_ids(
        catalogue_path
    )

    print("=" * 72)
    print("QLORA CHECKPOINT SELECTION")
    print("=" * 72)
    print(f"Model:              {model_name}")
    print(f"Checkpoints:        {len(checkpoints)}")
    print(f"Validation users:   {len(validation_records):,}")
    print(f"Catalogue items:    {len(catalogue_item_ids):,}")
    print(f"Top K:              {top_k}")
    print(f"Beam width:         {num_beams}")
    print(f"Batch size:         {batch_size}")
    print(f"Prompt history:     latest {max_history_length} items")
    print("History exclusion:  complete real history")
    print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(
            nested_value(
                config,
                "model.trust_remote_code",
                default=False,
            )
        ),
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    compute_dtype = select_compute_dtype(
        str(
            nested_value(
                config,
                "quantization.compute_dtype",
                default="auto",
            )
        )
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=bool(
            nested_value(
                config,
                "quantization.use_double_quant",
                default=True,
            )
        ),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("Loading base model in 4-bit...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        trust_remote_code=bool(
            nested_value(
                config,
                "model.trust_remote_code",
                default=False,
            )
        ),
        low_cpu_mem_usage=True,
        device_map={"": torch.cuda.current_device()},
        attn_implementation=nested_value(
            config,
            "model.attention_implementation",
            default="sdpa",
        ),
    )

    base_model.config.use_cache = True

    print("Building full-catalogue decoding constraint...")

    catalogue_constraint = CatalogueTokenConstraint(
        tokenizer=tokenizer,
        item_ids=catalogue_item_ids,
    )

    history_constraint = HistoryAwareConstraint(
        catalogue_constraint
    )

    selection_directory = (
        output_directory / "validation_selection"
    )
    selection_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = None
    checkpoint_results: list[dict[str, Any]] = []

    for checkpoint_index, checkpoint_path in enumerate(
        checkpoints
    ):
        adapter_name = f"checkpoint_{checkpoint_index}"

        print()
        print("-" * 72)
        print(f"Evaluating: {checkpoint_path.name}")
        print("-" * 72)

        if model is None:
            model = PeftModel.from_pretrained(
                base_model,
                str(checkpoint_path),
                adapter_name=adapter_name,
                is_trainable=False,
            )
        else:
            model.load_adapter(
                str(checkpoint_path),
                adapter_name=adapter_name,
                is_trainable=False,
            )

        model.set_adapter(adapter_name)
        model.eval()

        metrics, predictions = evaluate_adapter(
            model=model,
            tokenizer=tokenizer,
            validation_records=validation_records,
            metadata_lookup=metadata_lookup,
            catalogue_constraint=catalogue_constraint,
            history_constraint=history_constraint,
            max_history_length=max_history_length,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            top_k=top_k,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        prediction_path = (
            selection_directory
            / f"{checkpoint_path.name}_predictions.jsonl"
        )

        metric_path = (
            selection_directory
            / f"{checkpoint_path.name}_metrics.json"
        )

        write_jsonl(predictions, prediction_path)
        write_json(metrics, metric_path)

        checkpoint_result = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.name,
            "prediction_file": str(prediction_path),
            "metrics_file": str(metric_path),
            "metrics": metrics,
        }

        checkpoint_results.append(checkpoint_result)

        print(
            f"NDCG@{top_k}: "
            f"{metrics[f'ndcg@{top_k}']:.6f}"
        )
        print(
            f"Hit@{top_k}:  "
            f"{metrics[f'hit@{top_k}']:.6f}"
        )
        print(
            f"MRR@{top_k}:  "
            f"{metrics[f'mrr@{top_k}']:.6f}"
        )

        torch.cuda.empty_cache()

    checkpoint_results.sort(
        key=lambda result: (
            result["metrics"][f"ndcg@{top_k}"],
            result["metrics"][f"mrr@{top_k}"],
            result["metrics"][f"hit@{top_k}"],
        ),
        reverse=True,
    )

    best_result = checkpoint_results[0]

    selection_summary = {
        "experiment": nested_value(
            config,
            "experiment.name",
            required=True,
        ),
        "dataset": nested_value(
            config,
            "experiment.dataset",
            required=True,
        ),
        "base_model": model_name,
        "selection_metric": f"ndcg@{top_k}",
        "validation_users": len(validation_records),
        "validation_sampling_seed": seed,
        "full_validation_set_used": (
            max_validation_users is None
        ),
        "max_prompt_history_length": max_history_length,
        "history_exclusion_scope": "complete_real_history",
        "best_checkpoint": best_result["checkpoint"],
        "best_checkpoint_name": (
            best_result["checkpoint_name"]
        ),
        "best_metrics": best_result["metrics"],
        "all_checkpoints": checkpoint_results,
    }

    summary_path = (
        selection_directory / "checkpoint_selection.json"
    )

    write_json(selection_summary, summary_path)

    print()
    print("=" * 72)
    print("CHECKPOINT SELECTION COMPLETE")
    print("=" * 72)
    print(
        f"Best checkpoint: "
        f"{best_result['checkpoint_name']}"
    )
    print(
        f"Best NDCG@{top_k}: "
        f"{best_result['metrics'][f'ndcg@{top_k}']:.6f}"
    )
    print(f"Selection summary: {summary_path}")


if __name__ == "__main__":
    main()
