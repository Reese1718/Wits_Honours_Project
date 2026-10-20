"""
Step 06: Generate test-set recommendations using the checkpoint selected in
Step 05.

This script:

1. Reads the winning checkpoint from checkpoint_selection.json.
2. Loads the base model in 4-bit and attaches the selected QLoRA adapter.
3. Reads each user's test history (items 1 ... n-1).
4. Generates top-K catalogue-constrained recommendations.
5. Excludes items already present in the user's history.
6. Writes resumable JSONL predictions for Step 07.

It does NOT calculate final test metrics. That happens in Step 07.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


from common import build_recommendation_messages


# ---------------------------------------------------------------------------
# Prompt format
# ---------------------------------------------------------------------------
def build_chat_prompt(
    tokenizer: AutoTokenizer,
    history: list[str],
    metadata: dict[str, dict[str, Any]],
    max_history_length: int,
) -> str:
    """Apply the model's chat template to one recommendation prompt."""

    messages = build_recommendation_messages(
        history_item_ids=history,
        metadata_lookup=metadata,
        max_history_length=max_history_length,
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def nested_value(
    config: dict[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    value: Any = config

    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def dataset_name_from_config(config: dict[str, Any]) -> str:
    for key in ("experiment.dataset", "dataset.name"):
        value = nested_value(config, key)
        if value is not None and value != "":
            return str(value)

    dataset_value = config.get("dataset")
    if isinstance(dataset_value, (str, int, float, bool)):
        return str(dataset_value)

    if isinstance(dataset_value, dict) and dataset_value.get("dataset_name"):
        return str(dataset_value["dataset_name"])

    return "dataset"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def find_repository_root(config_path: Path) -> Path:
    """
    Locate the repository root by looking for pipeline/llm.
    """

    candidates = [
        Path.cwd().resolve(),
        config_path.resolve().parent,
        *config_path.resolve().parents,
    ]

    for candidate in candidates:
        if (candidate / "pipeline" / "llm").is_dir():
            return candidate

    return Path.cwd().resolve()


def resolve_path(
    value: str | Path,
    repository_root: Path,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (repository_root / path).resolve()


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} of {path} is not a JSON object."
                )

            records.append(record)

    return records


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")

    os.replace(temporary_path, path)


def repair_and_read_completed_indices(path: Path) -> set[int]:
    """
    Recover a prediction file after a possible Colab interruption.

    Valid JSONL rows are retained. A partially written final row is removed.
    """

    if not path.exists():
        return set()

    valid_lines: list[str] = []
    completed_indices: set[int] = set()
    encountered_invalid_line = False

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                print(
                    f"Warning: removing incomplete prediction line "
                    f"{line_number} from {path}"
                )
                encountered_invalid_line = True
                continue

            record_index = record.get("record_index")

            if record_index is None:
                raise ValueError(
                    f"Existing prediction line {line_number} does not contain "
                    f"'record_index'. Use --overwrite to start again."
                )

            record_index = int(record_index)

            if record_index in completed_indices:
                print(
                    f"Warning: ignoring duplicate record_index "
                    f"{record_index} in {path}"
                )
                encountered_invalid_line = True
                continue

            completed_indices.add(record_index)
            valid_lines.append(
                json.dumps(record, ensure_ascii=False)
            )

    if encountered_invalid_line:
        temporary_path = path.with_suffix(path.suffix + ".repair")

        with temporary_path.open("w", encoding="utf-8") as file:
            for line in valid_lines:
                file.write(line + "\n")

        os.replace(temporary_path, path)

    return completed_indices


# ---------------------------------------------------------------------------
# Dataset schema helpers
# ---------------------------------------------------------------------------

def first_present(
    record: dict[str, Any],
    keys: Iterable[str],
    required: bool = True,
) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]

    if required:
        raise KeyError(
            f"None of the required fields were found: {list(keys)}. "
            f"Available fields: {sorted(record.keys())}"
        )

    return None


def normalise_item_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return []

        # Attempt to interpret a JSON-formatted list.
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
                if isinstance(decoded, list):
                    return normalise_item_list(decoded)
            except json.JSONDecodeError:
                pass

        if "," in stripped:
            return [
                item.strip()
                for item in stripped.split(",")
                if item.strip()
            ]

        return stripped.split()

    raise TypeError(
        f"Expected an item list or string, received {type(value).__name__}."
    )


def test_record_values(
    record: dict[str, Any],
) -> tuple[str, list[str], str]:
    user_id = str(
        first_present(
            record,
            ["user_id", "user", "uid"],
        )
    )

    history = normalise_item_list(
        first_present(
            record,
            [
                "history",
                "history_item_ids",
                "item_history",
                "input_items",
                "sequence",
            ],
        )
    )

    target = str(
        first_present(
            record,
            [
                "target",
                "target_item_id",
                "next_item_id",
                "label",
                "test_item",
            ],
        )
    ).strip()

    if not history:
        raise ValueError(f"User {user_id} has an empty test history.")

    if not target:
        raise ValueError(f"User {user_id} has an empty test target.")

    return user_id, history, target


# ---------------------------------------------------------------------------
# Metadata and catalogue loading
# ---------------------------------------------------------------------------

def usable_title(record: Any) -> str:
    if not isinstance(record, dict):
        return ""

    value = record.get("title", "")

    if value is None:
        return ""

    return str(value).strip()


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    value = read_json(path)

    if isinstance(value, dict):
        # Standard lookup format:
        # {"ASIN": {"asin": "ASIN", "title": "..."}}
        if all(isinstance(record, dict) for record in value.values()):
            return {
                str(item_id): record
                for item_id, record in value.items()
            }

        # Wrapped lookup format.
        for key in ["metadata", "items", "records"]:
            wrapped = value.get(key)
            if isinstance(wrapped, list):
                value = wrapped
                break
            if isinstance(wrapped, dict):
                return {
                    str(item_id): record
                    for item_id, record in wrapped.items()
                    if isinstance(record, dict)
                }

    if isinstance(value, list):
        lookup: dict[str, dict[str, Any]] = {}

        for record in value:
            if not isinstance(record, dict):
                continue

            item_id = first_present(
                record,
                ["item_id", "asin", "id"],
                required=False,
            )

            if item_id is not None:
                lookup[str(item_id)] = record

        return lookup

    raise ValueError(f"Unsupported metadata structure in {path}")


def extract_catalogue_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        result: list[str] = []

        for entry in value:
            if isinstance(entry, str):
                result.append(entry)
            elif isinstance(entry, dict):
                item_id = first_present(
                    entry,
                    ["item_id", "asin", "id"],
                    required=False,
                )
                if item_id is not None:
                    result.append(str(item_id))

        return result

    if isinstance(value, dict):
        for key in [
            "item_ids",
            "catalogue",
            "catalog",
            "items",
            "records",
        ]:
            if key in value:
                result = extract_catalogue_ids(value[key])
                if result:
                    return result

        # Handle an item-ID-keyed dictionary.
        if value and all(
            isinstance(record, dict) for record in value.values()
        ):
            return [str(item_id) for item_id in value.keys()]

    raise ValueError("Could not identify item IDs in the catalogue file.")


def load_catalogue(path: Path) -> list[str]:
    catalogue = extract_catalogue_ids(read_json(path))

    # Preserve order while removing accidental duplicates.
    unique_catalogue = list(dict.fromkeys(
        item_id.strip()
        for item_id in catalogue
        if item_id.strip()
    ))

    if not unique_catalogue:
        raise ValueError(f"The item catalogue is empty: {path}")

    return unique_catalogue


# ---------------------------------------------------------------------------
# Catalogue token trie
# ---------------------------------------------------------------------------

@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    terminal_items: set[str] = field(default_factory=set)
    subtree_items: int = 0


class CatalogueTokenTrie:
    """
    Token trie for full-catalogue constrained decoding.

    subtree_items allows us to suppress branches containing only items that
    the current user has already interacted with, without rebuilding the
    catalogue trie for every user.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        item_ids: list[str],
    ) -> None:
        self.root = TrieNode()
        self.item_tokens: dict[str, tuple[int, ...]] = {}

        for item_id in item_ids:
            token_ids = tuple(
                tokenizer.encode(
                    item_id,
                    add_special_tokens=False,
                )
            )

            if not token_ids:
                raise ValueError(
                    f"Catalogue item {item_id!r} produced no tokens."
                )

            self.item_tokens[item_id] = token_ids
            self._insert(item_id, token_ids)

    def _insert(
        self,
        item_id: str,
        token_ids: tuple[int, ...],
    ) -> None:
        node = self.root
        node.subtree_items += 1

        for token_id in token_ids:
            node = node.children.setdefault(token_id, TrieNode())
            node.subtree_items += 1

        node.terminal_items.add(item_id)

    def allowed_tokens(
        self,
        generated_tokens: tuple[int, ...],
        excluded_items: set[str],
        eos_token_id: int,
    ) -> list[int]:
        node = self.root

        for token_id in generated_tokens:
            child = node.children.get(token_id)

            if child is None:
                return [eos_token_id]

            node = child

        allowed: list[int] = []

        # EOS is allowed only when the current sequence represents at least
        # one catalogue item that is not in the user's history.
        if any(
            item_id not in excluded_items
            for item_id in node.terminal_items
        ):
            allowed.append(eos_token_id)

        prefix_length = len(generated_tokens)

        for next_token, child in node.children.items():
            excluded_in_branch = 0

            for item_id in excluded_items:
                token_sequence = self.item_tokens.get(item_id)

                if token_sequence is None:
                    continue

                if len(token_sequence) <= prefix_length:
                    continue

                if (
                    token_sequence[:prefix_length] == generated_tokens
                    and token_sequence[prefix_length] == next_token
                ):
                    excluded_in_branch += 1

            # Keep the branch if it contains at least one unseen item.
            if child.subtree_items > excluded_in_branch:
                allowed.append(next_token)

        if not allowed:
            # This should only occur if every possible continuation has been
            # excluded. Returning EOS prevents Transformers from failing.
            return [eos_token_id]

        return allowed


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------

def recursive_find_checkpoint(value: Any) -> str | None:
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        preferred_keys = [
            "best_checkpoint",
            "selected_checkpoint",
            "checkpoint_path",
            "adapter_path",
            "path",
        ]

        for key in preferred_keys:
            if key in value:
                result = recursive_find_checkpoint(value[key])
                if result:
                    return result

        for nested in value.values():
            result = recursive_find_checkpoint(nested)
            if result and "checkpoint" in result.lower():
                return result

    return None


def resolve_selected_checkpoint(
    selection_file: Path,
    repository_root: Path,
    output_directory: Path,
) -> Path:
    selection = read_json(selection_file)
    checkpoint_value = recursive_find_checkpoint(selection)

    if not checkpoint_value:
        raise ValueError(
            f"Could not find the selected checkpoint in {selection_file}"
        )

    raw_path = Path(checkpoint_value)

    candidates = [
        raw_path,
        repository_root / raw_path,
        output_directory / raw_path,
        selection_file.parent / raw_path,
    ]

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate.is_dir():
            return candidate

    attempted = "\n".join(
        f"  - {candidate.resolve()}" for candidate in candidates
    )

    raise FileNotFoundError(
        f"Selected checkpoint directory was not found.\n"
        f"Checkpoint value: {checkpoint_value}\n"
        f"Attempted:\n{attempted}"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def choose_compute_dtype(value: str) -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was not detected. Step 06 must be run on a GPU compute "
            "node."
        )

    requested = value.lower()

    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    if requested in {"bfloat16", "bf16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "The selected GPU does not support bfloat16. Use float16."
            )

        return torch.bfloat16

    if requested in {"float16", "fp16"}:
        return torch.float16

    raise ValueError(
        "quantization.compute_dtype must be auto, bfloat16, or float16."
    )


def configure_tokenizer(
    tokenizer: AutoTokenizer,
    config: dict[str, Any],
) -> int:
    configured_eos_token = nested_value(
        config,
        "model.eos_token",
        default=None,
    )

    if configured_eos_token:
        configured_id = tokenizer.convert_tokens_to_ids(
            configured_eos_token
        )

        if (
            configured_id is not None
            and configured_id != tokenizer.unk_token_id
        ):
            tokenizer.eos_token = configured_eos_token

    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer does not define an EOS token.")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    return int(tokenizer.eos_token_id)


def load_model_and_tokenizer(
    model_name: str,
    checkpoint_path: Path,
    config: dict[str, Any],
) -> tuple[PeftModel, AutoTokenizer, torch.dtype]:
    compute_dtype = choose_compute_dtype(
        str(
            nested_value(
                config,
                "quantization.compute_dtype",
                default="auto",
            )
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(
            nested_value(
                config,
                "model.trust_remote_code",
                default=False,
            )
        ),
    )

    configure_tokenizer(tokenizer, config)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(
            nested_value(
                config,
                "quantization.quant_type",
                default="nf4",
            )
        ),
        bnb_4bit_use_double_quant=bool(
            nested_value(
                config,
                "quantization.use_double_quant",
                default=True,
            )
        ),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={"": torch.cuda.current_device()},
        dtype=compute_dtype,
        trust_remote_code=bool(
            nested_value(
                config,
                "model.trust_remote_code",
                default=False,
            )
        ),
        attn_implementation=nested_value(
            config,
            "model.attention_implementation",
            default="sdpa",
        ),
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_path),
        is_trainable=False,
    )

    model.eval()
    model.config.use_cache = True

    return model, tokenizer, compute_dtype


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_batch(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    trie: CatalogueTokenTrie,
    catalogue_set: set[str],
    records: list[tuple[int, dict[str, Any]]],
    metadata: dict[str, dict[str, Any]],
    max_history_length: int,
    top_k: int,
    num_beams: int,
    max_new_tokens: int,
    maximum_input_length: int,
    exclude_history_items: bool,
    length_penalty: float,
) -> list[dict[str, Any]]:
    prompts: list[str] = []
    parsed_records: list[tuple[int, str, list[str], str]] = []
    excluded_by_user: list[set[str]] = []

    for record_index, record in records:
        user_id, history, target = test_record_values(record)

        prompts.append(
            build_chat_prompt(
                tokenizer=tokenizer,
                history=history,
                metadata=metadata,
                max_history_length=max_history_length,
            )
        )

        parsed_records.append(
            (record_index, user_id, history, target)
        )

        if exclude_history_items:
            excluded_by_user.append(set(history))
        else:
            excluded_by_user.append(set())

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=maximum_input_length,
        add_special_tokens=False,
    )

    input_width = int(encoded["input_ids"].shape[1])
    model_device = next(model.parameters()).device

    encoded = {
        key: value.to(model_device)
        for key, value in encoded.items()
    }

    eos_token_id = int(tokenizer.eos_token_id)

    def prefix_allowed_tokens_fn(
        batch_id: int,
        input_ids: torch.Tensor,
    ) -> list[int]:
        generated = tuple(
            int(token_id)
            for token_id in input_ids[input_width:].tolist()
        )

        return trie.allowed_tokens(
            generated_tokens=generated,
            excluded_items=excluded_by_user[int(batch_id)],
            eos_token_id=eos_token_id,
        )

    beam_count = max(num_beams, top_k)
    return_sequence_count = top_k

    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=beam_count,
        num_return_sequences=return_sequence_count,
        early_stopping=True,
        length_penalty=length_penalty,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        return_dict_in_generate=True,
        output_scores=True,
    )

    sequences = generated.sequences
    sequence_scores = generated.sequences_scores

    batch_results: list[dict[str, Any]] = []

    for batch_position, parsed in enumerate(parsed_records):
        record_index, user_id, history, target = parsed
        excluded_items = excluded_by_user[batch_position]

        start = batch_position * return_sequence_count
        end = start + return_sequence_count

        ranked_candidates: list[tuple[str, float]] = []
        already_added: set[str] = set()

        for sequence_position in range(start, end):
            generated_tokens = sequences[
                sequence_position,
                input_width:,
            ]

            prediction = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            # Guard against occasional formatting despite constrained
            # generation.
            prediction = prediction.splitlines()[0].strip()
            prediction = prediction.strip("`\"' ")

            if prediction not in catalogue_set:
                continue

            if prediction in excluded_items:
                continue

            if prediction in already_added:
                continue

            score = float(sequence_scores[sequence_position].item())

            ranked_candidates.append((prediction, score))
            already_added.add(prediction)

            if len(ranked_candidates) == top_k:
                break

        recommendations = [
            item_id for item_id, _ in ranked_candidates
        ]

        scores = [
            score for _, score in ranked_candidates
        ]

        batch_results.append(
            {
                "record_index": record_index,
                "user_id": user_id,
                "history": history,
                "target": target,
                "recommendations": recommendations,
                "scores": scores,
                "requested_top_k": top_k,
                "generated_recommendation_count": len(recommendations),
            }
        )

    return batch_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate full-catalogue test recommendations using the "
            "validation-selected QLoRA checkpoint."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment YAML configuration.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing Step 06 predictions and start again.",
    )

    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help=(
            "Optional debugging limit. Do not use this for the final "
            "experiment."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment.seed from the YAML configuration.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config_path = args.config.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    config = load_yaml(config_path)
    repository_root = find_repository_root(config_path)

    seed = (
        args.seed
        if args.seed is not None
        else int(nested_value(config, "experiment.seed", default=42))
    )
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset_name = dataset_name_from_config(config)

    model_name = nested_value(
        config,
        "model.name_or_path",
        default=nested_value(config, "model.name"),
    )

    if not model_name:
        raise ValueError(
            "The YAML file must contain model.name_or_path."
        )

    output_directory_value = nested_value(
        config,
        "training.output_dir",
    )

    if not output_directory_value:
        raise ValueError(
            "The YAML file must contain training.output_dir."
        )

    output_directory = resolve_path(
        output_directory_value,
        repository_root,
    )

    test_file = resolve_path(
        nested_value(config, "data.test_split_file"),
        repository_root,
    )

    metadata_file = resolve_path(
        nested_value(config, "data.metadata_file"),
        repository_root,
    )

    catalogue_file = resolve_path(
        nested_value(config, "data.catalogue_file"),
        repository_root,
    )

    selection_file_value = nested_value(
        config,
        "generation.checkpoint_selection_file",
        default=(
            output_directory
            / "validation_selection"
            / "checkpoint_selection.json"
        ),
    )

    selection_file = resolve_path(
        selection_file_value,
        repository_root,
    )

    predictions_file_value = nested_value(
        config,
        "generation.output_file",
        default=(
            output_directory
            / "test_predictions"
            / "test_predictions.jsonl"
        ),
    )

    predictions_file = Path(predictions_file_value)

    if not predictions_file.is_absolute():
        predictions_file = resolve_path(
            predictions_file,
            repository_root,
        )

    summary_file = predictions_file.parent / "generation_summary.json"

    required_files = [
        test_file,
        metadata_file,
        catalogue_file,
        selection_file,
    ]

    missing_files = [
        path for path in required_files if not path.is_file()
    ]

    if missing_files:
        formatted = "\n".join(
            f"  - {path}" for path in missing_files
        )
        raise FileNotFoundError(
            f"Required Step 06 inputs are missing:\n{formatted}"
        )

    selected_checkpoint = resolve_selected_checkpoint(
        selection_file=selection_file,
        repository_root=repository_root,
        output_directory=output_directory,
    )

    top_k = int(
        nested_value(config, "generation.top_k", default=10)
    )

    num_beams = int(
        nested_value(config, "generation.num_beams", default=10)
    )

    batch_size = int(
        nested_value(config, "generation.batch_size", default=4)
    )

    max_new_tokens = int(
        nested_value(
            config,
            "generation.max_new_tokens",
            default=24,
        )
    )

    max_history_length = int(
        nested_value(
            config,
            "generation.max_history_length",
            default=nested_value(
                config,
                "checkpoint_selection.max_history_length",
                default=5,
            ),
        )
    )

    maximum_sequence_length = int(
        nested_value(
            config,
            "training.max_sequence_length",
            default=nested_value(
                config,
                "training.max_length",
                default=nested_value(
                    config,
                    "training.max_seq_length",
                    default=2048,
                ),
            ),
        )
    )

    maximum_input_length = (
        maximum_sequence_length - max_new_tokens
    )

    exclude_history_items = bool(
        nested_value(
            config,
            "generation.exclude_history_items",
            default=True,
        )
    )

    length_penalty = float(
        nested_value(
            config,
            "generation.length_penalty",
            default=1.0,
        )
    )

    if top_k <= 0:
        raise ValueError("generation.top_k must be positive.")

    if num_beams < top_k:
        raise ValueError(
            "generation.num_beams must be greater than or equal to top_k."
        )

    if batch_size <= 0:
        raise ValueError("generation.batch_size must be positive.")

    if max_history_length <= 0:
        raise ValueError(
            "generation.max_history_length must be positive."
        )

    if max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be positive.")

    if maximum_input_length <= 0:
        raise ValueError(
            "The configured maximum sequence length must be greater than "
            "generation.max_new_tokens."
        )

    selection_summary = read_json(selection_file)
    selected_history_length = selection_summary.get(
        "max_prompt_history_length"
    )

    if (
        selected_history_length is not None
        and int(selected_history_length) != max_history_length
    ):
        raise ValueError(
            "Step 05 and Step 06 use different prompt-history lengths: "
            f"Step 05 used {selected_history_length}, but Step 06 is "
            f"configured for {max_history_length}."
        )

    print("=" * 70)
    print("GENERATING TEST RECOMMENDATIONS")
    print("=" * 70)
    print(f"Dataset:             {dataset_name}")
    print(f"Base model:          {model_name}")
    print(f"Selected checkpoint: {selected_checkpoint}")
    print(f"Test file:           {test_file}")
    print(f"Catalogue file:      {catalogue_file}")
    print(f"Predictions file:    {predictions_file}")
    print(f"Top-K:               {top_k}")
    print(f"Beams:               {num_beams}")
    print(f"Batch size:          {batch_size}")
    print(f"Prompt history:      latest {max_history_length} items")
    print(f"Exclude history:     {exclude_history_items}")
    if exclude_history_items:
        print("Exclusion scope:     complete real test history")
    print()

    print("Loading test records...")
    test_records = read_jsonl(test_file)

    if args.max_users is not None:
        if args.max_users <= 0:
            raise ValueError("--max-users must be positive.")

        test_records = test_records[: args.max_users]

        print(
            f"Debug limit enabled: using the first "
            f"{len(test_records):,} users."
        )

    print(f"Test records: {len(test_records):,}")

    print("Loading metadata...")
    metadata = load_metadata(metadata_file)
    print(f"Metadata records: {len(metadata):,}")

    print("Loading item catalogue...")
    catalogue = load_catalogue(catalogue_file)
    catalogue_set = set(catalogue)
    print(f"Catalogue items: {len(catalogue):,}")

    print("Checking test records...")

    for record_index, record in enumerate(test_records):
        user_id, history, target = test_record_values(record)

        unknown_history = [
            item_id
            for item_id in history
            if item_id not in catalogue_set
        ]

        if unknown_history:
            raise ValueError(
                f"User {user_id} has history items that are not in the "
                f"catalogue: {unknown_history[:10]}"
            )

        if target not in catalogue_set:
            raise ValueError(
                f"Test target {target!r} for user {user_id} is not in "
                f"the catalogue."
            )

    predictions_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.overwrite:
        if predictions_file.exists():
            predictions_file.unlink()

        if summary_file.exists():
            summary_file.unlink()

    completed_indices = repair_and_read_completed_indices(
        predictions_file
    )

    pending_records = [
        (record_index, record)
        for record_index, record in enumerate(test_records)
        if record_index not in completed_indices
    ]

    print(f"Already completed: {len(completed_indices):,}")
    print(f"Remaining records: {len(pending_records):,}")

    if not pending_records:
        print("All requested test records have already been generated.")
        print(f"Predictions: {predictions_file}")
        return

    print()
    print("Loading tokenizer and selected QLoRA adapter...")

    model, tokenizer, compute_dtype = load_model_and_tokenizer(
        model_name=model_name,
        checkpoint_path=selected_checkpoint,
        config=config,
    )

    print(f"Compute dtype: {compute_dtype}")
    print("Building catalogue token trie...")

    trie = CatalogueTokenTrie(
        tokenizer=tokenizer,
        item_ids=catalogue,
    )

    print("Catalogue trie ready.")
    print()

    generation_started = time.time()
    generated_this_run = 0
    short_recommendation_rows = 0

    with predictions_file.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as output_file:
        progress = tqdm(
            range(0, len(pending_records), batch_size),
            desc="Generating test recommendations",
            unit="batch",
        )

        for start in progress:
            batch = pending_records[start : start + batch_size]

            batch_results = generate_batch(
                model=model,
                tokenizer=tokenizer,
                trie=trie,
                catalogue_set=catalogue_set,
                records=batch,
                metadata=metadata,
                max_history_length=max_history_length,
                top_k=top_k,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                maximum_input_length=maximum_input_length,
                exclude_history_items=exclude_history_items,
                length_penalty=length_penalty,
            )

            for result in batch_results:
                if (
                    result["generated_recommendation_count"]
                    < top_k
                ):
                    short_recommendation_rows += 1

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            output_file.flush()

            generated_this_run += len(batch_results)

            progress.set_postfix(
                completed=(
                    len(completed_indices)
                    + generated_this_run
                ),
                short_rows=short_recommendation_rows,
            )

    elapsed_seconds = time.time() - generation_started
    total_completed = (
        len(completed_indices) + generated_this_run
    )

    summary = {
        "dataset": dataset_name,
        "config_file": str(config_path),
        "base_model": model_name,
        "selected_checkpoint": str(selected_checkpoint),
        "checkpoint_selection_file": str(selection_file),
        "test_split_file": str(test_file),
        "catalogue_file": str(catalogue_file),
        "metadata_file": str(metadata_file),
        "predictions_file": str(predictions_file),
        "test_record_count": len(test_records),
        "completed_record_count": total_completed,
        "generated_this_run": generated_this_run,
        "requested_top_k": top_k,
        "num_beams": num_beams,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "max_history_length": max_history_length,
        "maximum_input_length": maximum_input_length,
        "exclude_history_items": exclude_history_items,
        "short_recommendation_rows_this_run": (
            short_recommendation_rows
        ),
        "elapsed_seconds_this_run": elapsed_seconds,
        "records_per_second_this_run": (
            generated_this_run / elapsed_seconds
            if elapsed_seconds > 0
            else None
        ),
        "history_exclusion_scope": (
            "complete_real_history"
            if exclude_history_items
            else "none"
        ),
        "seed": seed,
        "debug_max_users": args.max_users,
    }

    atomic_write_json(summary_file, summary)

    print()
    print("=" * 70)
    print("TEST RECOMMENDATION GENERATION COMPLETE")
    print("=" * 70)
    print(f"Completed records: {total_completed:,}")
    print(f"Predictions:       {predictions_file}")
    print(f"Summary:           {summary_file}")
    print(
        f"Elapsed this run:  {elapsed_seconds / 60:.2f} minutes"
    )

    if short_recommendation_rows:
        print(
            f"Warning: {short_recommendation_rows:,} rows generated "
            f"fewer than {top_k} unique recommendations."
        )

    print()
    print(
        "The true test targets have not been scored yet. "
        "Run Step 07 to calculate the final test metrics."
    )


if __name__ == "__main__":
    main()
