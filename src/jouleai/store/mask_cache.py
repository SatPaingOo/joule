"""Activation mask cache: map query -> per-layer FFN keep-masks.

The "database index" of the paradigm: serve a query similar to a previous one by
reusing its neuron mask instead of loading/computing the full FFN.
"""

from __future__ import annotations

import torch


def embed_query(tokenizer, model, query: str, device: str = "cpu") -> torch.Tensor:
    """Mean-pooled final hidden state as the query signature (L2-normalised)."""
    ids = tokenizer(query, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    h = out.hidden_states[-1][0].mean(dim=0).float()
    return h / h.norm()


class ActivationMaskCache:
    """Nearest-neighbour lookup of stored masks keyed by query signature."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        self.keys: list[torch.Tensor] = []
        self.masks: list[list[torch.Tensor]] = []
        self.queries: list[str] = []

    def put(self, sig: torch.Tensor, masks: list[torch.Tensor], query: str) -> None:
        self.keys.append(sig.cpu())
        self.masks.append([m.cpu() for m in masks])
        self.queries.append(query)

    def get(self, sig: torch.Tensor) -> tuple[int, float] | None:
        """Return (index, similarity) of best match above threshold, else None."""
        if not self.keys:
            return None
        K = torch.stack(self.keys)
        sims = K @ sig.cpu().float()
        best_sim, best_i = sims.max(dim=0)
        if float(best_sim) >= self.threshold:
            return int(best_i), float(best_sim)
        return None

    def masks_at(self, i: int) -> list[torch.Tensor]:
        return self.masks[i]

    def __len__(self) -> int:
        return len(self.keys)
