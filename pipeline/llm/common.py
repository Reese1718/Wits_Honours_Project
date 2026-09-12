"""
Shared utilities for the QLoRA recommender pipeline.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import yaml


# common.py is located at:
# <project_root>/pipeline/llm_recommender/common.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


SYSTEM_PROMPT = (
    "You are a sequential recommendation system. "
    "Given a user's interactions in chronological order, predict the "
    "single most likely item that the user will interact with next. "
    "Return only the item ID and no explanation."
)


def resolve_project_path(path: str | Path) -> Path:
    """
    Convert a project-relative path into an absolute path.

    Absolute paths are returned unchanged.
    """

    resolved = Path(path).expanduser()

    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved

    return resolved.resolve()


def ensure_parent_directory(path: str | Path) -> Path:
    """Create the parent directory of a file if it does not exist."""

    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    resolved = resolve_project_path(path)

    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected YAML configuration to contain a dictionary: {resolved}"
        )

    return config


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""

    resolved = resolve_project_path(path)

    if not resolved.exists():
        raise FileNotFoundError(f"JSON file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(
    data: Any,
    path: str | Path,
    indent: int = 2,
) -> Path:
    """
    Write JSON safely through a temporary file.

    The original output file is replaced only after writing succeeds.
    """

    resolved = ensure_parent_directory(path)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=indent,
        )

    temporary.replace(resolved)
    return resolved


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """
    Stream records from a JSONL file.

    Streaming avoids loading large datasets fully into memory.
    """

    resolved = resolve_project_path(path)

    if not resolved.exists():
        raise FileNotFoundError(f"JSONL file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {resolved}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {resolved}"
                )

            yield record


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load all records from a JSONL file."""

    return list(iter_jsonl(path))


def write_jsonl(
    records: Iterable[dict[str, Any]],
    path: str | Path,
) -> Path:
    """
    Write records to JSONL safely through a temporary file.
    """

    resolved = ensure_parent_directory(path)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    temporary.replace(resolved)
    return resolved


def set_random_seed(seed: int) -> None:
    """Set reproducible random seeds for Python, NumPy and PyTorch."""

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_recbole_header(header_line: str) -> list[str]:
    """
    Extract plain field names from a RecBole header.

    Example:
        user_id:token    item_id:token    timestamp:float

    Becomes:
        ["user_id", "item_id", "timestamp"]
    """

    typed_fields = header_line.rstrip("\n\r").split("\t")
    return [field.split(":", maxsplit=1)[0] for field in typed_fields]


def iter_recbole_interactions(
    path: str | Path,
    user_field: str = "user_id",
    item_field: str = "item_id",
    time_field: str = "timestamp",
) -> Iterator[dict[str, Any]]:
    """
    Stream interactions from a RecBole .inter file.

    Returned records use the canonical keys:
        user_id
        item_id
        timestamp
        row_number

    row_number provides deterministic ordering when timestamps are equal.
    """

    resolved = resolve_project_path(path)

    if not resolved.exists():
        raise FileNotFoundError(f"RecBole interaction file not found: {resolved}")

    with resolved.open("r", encoding="utf-8-sig") as file:
        header_line = file.readline()

        if not header_line:
            raise ValueError(f"Interaction file is empty: {resolved}")

        fields = parse_recbole_header(header_line)

        required_fields = {user_field, item_field, time_field}
        missing_fields = required_fields.difference(fields)

        if missing_fields:
            raise ValueError(
                f"Missing required fields in {resolved}: "
                f"{sorted(missing_fields)}. Found: {fields}"
            )

        user_index = fields.index(user_field)
        item_index = fields.index(item_field)
        time_index = fields.index(time_field)

        for row_number, line in enumerate(file, start=2):
            line = line.rstrip("\n\r")

            if not line:
                continue

            values = line.split("\t")

            if len(values) != len(fields):
                raise ValueError(
                    f"Line {row_number} of {resolved} has {len(values)} values, "
                    f"but the header has {len(fields)} fields."
                )

            user_id = values[user_index].strip()
            item_id = values[item_index].strip()
            timestamp_text = values[time_index].strip()

            if not user_id:
                raise ValueError(
                    f"Missing user ID on line {row_number} of {resolved}"
                )

            if not item_id:
                raise ValueError(
                    f"Missing item ID on line {row_number} of {resolved}"
                )

            try:
                timestamp = float(timestamp_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid timestamp '{timestamp_text}' on line "
                    f"{row_number} of {resolved}"
                ) from error

            yield {
                "user_id": user_id,
                "item_id": item_id,
                "timestamp": timestamp,
                "row_number": row_number,
            }


def load_metadata_lookup(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    """
    Load an item metadata lookup.

    Supported structures:

    1. A dictionary keyed by item ID.
    2. A list of metadata objects containing item_id, parent_asin or asin.
    """

    raw_metadata = read_json(path)

    if isinstance(raw_metadata, dict):
        return {
            str(item_id): (
                metadata if isinstance(metadata, dict) else {"title": str(metadata)}
            )
            for item_id, metadata in raw_metadata.items()
        }

    if isinstance(raw_metadata, list):
        lookup: dict[str, dict[str, Any]] = {}

        possible_id_fields = (
            "item_id",
            "parent_asin",
            "asin",
        )

        for position, metadata in enumerate(raw_metadata):
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Metadata record {position} is not a JSON object."
                )

            item_id = next(
                (
                    metadata.get(field)
                    for field in possible_id_fields
                    if metadata.get(field)
                ),
                None,
            )

            if item_id is None:
                raise ValueError(
                    f"Metadata record {position} has no supported item-ID field."
                )

            lookup[str(item_id)] = metadata

        return lookup

    raise ValueError(
        "Metadata must be either a dictionary keyed by item ID "
        "or a list of metadata objects."
    )


def clean_text(value: Any) -> str:
    """Normalize metadata text for use inside a one-line prompt."""

    if value is None:
        return ""

    return " ".join(str(value).split())


def get_item_title(
    item_id: str,
    metadata_lookup: dict[str, dict[str, Any]],
) -> str:
    """Return an item's title, with a safe fallback."""

    metadata = metadata_lookup.get(str(item_id), {})

    for field in ("title", "name", "product_title"):
        title = clean_text(metadata.get(field))

        if title:
            return title

    return "Unknown title"


def format_history(
    history_item_ids: Sequence[str],
    metadata_lookup: dict[str, dict[str, Any]],
    max_history_length: int = 50,
) -> str:
    """
    Format a chronological interaction history for the LLM.

    If a history exceeds the configured limit, the most recent items are used.
    """

    if max_history_length <= 0:
        raise ValueError("max_history_length must be greater than zero.")

    truncated_history = list(history_item_ids)[-max_history_length:]

    if not truncated_history:
        raise ValueError("Cannot format an empty interaction history.")

    lines = []

    for position, item_id in enumerate(truncated_history, start=1):
        item_id = str(item_id)
        title = get_item_title(item_id, metadata_lookup)
        lines.append(f"{position}. {item_id} | {title}")

    return "\n".join(lines)


def build_recommendation_messages(
    history_item_ids: Sequence[str],
    metadata_lookup: dict[str, dict[str, Any]],
    max_history_length: int = 50,
) -> list[dict[str, str]]:
    """
    Build model-independent chat messages.

    The model-specific chat template will later be applied by its tokenizer.
    """

    formatted_history = format_history(
        history_item_ids=history_item_ids,
        metadata_lookup=metadata_lookup,
        max_history_length=max_history_length,
    )

    user_prompt = (
        "The user interacted with the following items in chronological order, "
        "from oldest to most recent:\n\n"
        f"{formatted_history}\n\n"
        "Predict the single item that the user is most likely to interact "
        "with next. Return only the item ID."
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def require_fields(
    record: dict[str, Any],
    required_fields: Sequence[str],
    context: str = "record",
) -> None:
    """Check that a dictionary contains all required fields."""

    missing = [
        field
        for field in required_fields
        if field not in record
    ]

    if missing:
        raise ValueError(
            f"Missing required fields in {context}: {missing}"
        )