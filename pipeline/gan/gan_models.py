import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        n_items,
        embedding_size=64,
        hidden_size=64,
    ):
        super().__init__()

        # Item ID 0 is reserved for padding.
        self.item_embedding = nn.Embedding(
            n_items,
            embedding_size,
            padding_idx=0,
        )

        self.gru = nn.GRU(
            embedding_size,
            hidden_size,
            batch_first=True,
        )

    def encode(self, history, lengths):
        packed = pack_padded_sequence(
            self.item_embedding(history),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed)

        return hidden[-1]


class Generator(HistoryEncoder):
    def __init__(
        self,
        n_items,
        embedding_size=64,
        hidden_size=64,
    ):
        super().__init__(
            n_items,
            embedding_size,
            hidden_size,
        )

        self.query = nn.Linear(
            hidden_size,
            embedding_size,
        )

        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, history, lengths, candidates):
        query = self.query(
            self.encode(history, lengths)
        )

        embeddings = self.item_embedding(candidates)

        scores = (
            query[:, None, :] * embeddings
        ).sum(-1)

        scores = scores / math.sqrt(
            embeddings.shape[-1]
        )

        return scores + self.item_bias(
            candidates
        ).squeeze(-1)


class Discriminator(HistoryEncoder):
    def __init__(
        self,
        n_items,
        embedding_size=64,
        hidden_size=64,
    ):
        super().__init__(
            n_items,
            embedding_size,
            hidden_size,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size + embedding_size,
                hidden_size,
            ),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        history,
        lengths,
        candidate_embedding,
    ):
        context = self.encode(history, lengths)

        combined = torch.cat(
            [context, candidate_embedding],
            dim=-1,
        )

        return self.classifier(
            combined
        ).squeeze(-1)