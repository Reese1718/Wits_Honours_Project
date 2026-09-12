"""
Catalogue-constrained decoding for next-item recommendation.

The LLM is permitted to generate only item IDs that exist in the
dataset catalogue. This prevents hallucinated or malformed item IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch


@dataclass
class TrieNode:
    """A single node in a token trie."""

    children: dict[int, "TrieNode"] = field(default_factory=dict)
    terminal: bool = False


class TokenTrie:
    """
    Trie containing tokenized catalogue item IDs.
    """

    def __init__(self) -> None:
        self.root = TrieNode()
        self.sequence_count = 0

    def insert(self, token_ids: Sequence[int]) -> None:
        """Insert one token sequence into the trie."""

        if not token_ids:
            raise ValueError("Cannot insert an empty token sequence.")

        node = self.root

        for token_id in token_ids:
            token_id = int(token_id)

            if token_id not in node.children:
                node.children[token_id] = TrieNode()

            node = node.children[token_id]

        if not node.terminal:
            node.terminal = True
            self.sequence_count += 1

    def get_node(self, prefix: Sequence[int]) -> TrieNode | None:
        """Return the trie node reached by a token prefix."""

        node = self.root

        for token_id in prefix:
            node = node.children.get(int(token_id))

            if node is None:
                return None

        return node

    def allowed_next_tokens(
        self,
        prefix: Sequence[int],
        eos_token_id: int,
    ) -> list[int]:
        """
        Return valid next tokens for the supplied generated prefix.
        """

        node = self.get_node(prefix)

        if node is None:
            return [eos_token_id]

        allowed = list(node.children.keys())

        if node.terminal:
            allowed.append(eos_token_id)

        if not allowed:
            return [eos_token_id]

        return allowed


class CatalogueTokenConstraint:
    """
    Tokenize item IDs and construct a full-catalogue decoding constraint.

    The object also stores a reverse mapping from generated token
    sequences back to the original item IDs.
    """

    def __init__(
        self,
        tokenizer,
        item_ids: Iterable[str],
        eos_token_id: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer

        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id

        if eos_token_id is None:
            raise ValueError(
                "The tokenizer has no EOS token. Supply eos_token_id explicitly."
            )

        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = tokenizer.pad_token_id

        self.trie = TokenTrie()
        self.token_sequence_to_item: dict[tuple[int, ...], str] = {}
        self.item_to_token_sequence: dict[str, tuple[int, ...]] = {}

        unique_item_ids = sorted({str(item_id).strip() for item_id in item_ids})

        if not unique_item_ids:
            raise ValueError("The item catalogue is empty.")

        for item_id in unique_item_ids:
            if not item_id:
                raise ValueError("The item catalogue contains an empty item ID.")

            token_ids = tokenizer.encode(
                item_id,
                add_special_tokens=False,
            )

            if not token_ids:
                raise ValueError(
                    f"Tokenizer produced no tokens for item ID: {item_id}"
                )

            token_sequence = tuple(int(token_id) for token_id in token_ids)

            existing_item = self.token_sequence_to_item.get(token_sequence)

            if existing_item is not None and existing_item != item_id:
                raise ValueError(
                    "Two item IDs produced the same token sequence: "
                    f"'{existing_item}' and '{item_id}'."
                )

            self.trie.insert(token_sequence)
            self.token_sequence_to_item[token_sequence] = item_id
            self.item_to_token_sequence[item_id] = token_sequence

    def __len__(self) -> int:
        return len(self.item_to_token_sequence)

    def build_prefix_allowed_tokens_fn(
        self,
        prompt_length: int,
    ) -> Callable[[int, torch.Tensor], list[int]]:
        """
        Create the callback expected by transformers.generate().

        This pipeline generates recommendations for one prompt at a time.
        prompt_length must therefore be the token length of that prompt.
        """

        if prompt_length <= 0:
            raise ValueError("prompt_length must be greater than zero.")

        def prefix_allowed_tokens_fn(
            batch_id: int,
            input_ids: torch.Tensor,
        ) -> list[int]:
            del batch_id

            generated_tokens = input_ids[prompt_length:].tolist()

            if self.eos_token_id in generated_tokens:
                eos_position = generated_tokens.index(self.eos_token_id)
                generated_tokens = generated_tokens[:eos_position]

            return self.trie.allowed_next_tokens(
                prefix=generated_tokens,
                eos_token_id=self.eos_token_id,
            )

        return prefix_allowed_tokens_fn

    def generated_tokens_to_item(
        self,
        generated_tokens: Sequence[int],
    ) -> str | None:
        """
        Convert a generated token sequence into its catalogue item ID.
        """

        cleaned_tokens: list[int] = []

        for token_id in generated_tokens:
            token_id = int(token_id)

            if token_id == self.eos_token_id:
                break

            if (
                self.pad_token_id is not None
                and token_id == self.pad_token_id
            ):
                continue

            cleaned_tokens.append(token_id)

        return self.token_sequence_to_item.get(tuple(cleaned_tokens))


def extract_ranked_items(
    generation_output,
    constraint: CatalogueTokenConstraint,
    prompt_length: int,
    top_k: int,
) -> list[dict[str, float | str | int]]:
    """
    Extract ranked, unique item IDs from beam-generation output.

    generation_output must be returned with:
        return_dict_in_generate=True
        output_scores=True
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    sequences = generation_output.sequences
    sequence_scores = getattr(
        generation_output,
        "sequences_scores",
        None,
    )

    candidates: list[tuple[str, float]] = []

    for index, sequence in enumerate(sequences):
        generated_tokens = sequence[prompt_length:].tolist()

        item_id = constraint.generated_tokens_to_item(generated_tokens)

        if item_id is None:
            continue

        if sequence_scores is None:
            score = float(-index)
        else:
            score = float(sequence_scores[index].item())

        candidates.append((item_id, score))

    candidates.sort(
        key=lambda candidate: candidate[1],
        reverse=True,
    )

    ranked_items: list[dict[str, float | str | int]] = []
    seen_items: set[str] = set()

    for item_id, score in candidates:
        if item_id in seen_items:
            continue

        seen_items.add(item_id)

        ranked_items.append(
            {
                "rank": len(ranked_items) + 1,
                "item_id": item_id,
                "score": score,
            }
        )

        if len(ranked_items) == top_k:
            break

    return ranked_items