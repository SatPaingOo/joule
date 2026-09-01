"""Mask policies: decide which FFN neurons to keep, training-free."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class MaskPolicy(ABC):
    """Given a per-layer activation vector, return a boolean keep-mask."""

    @abstractmethod
    def mask(self, act: torch.Tensor) -> torch.Tensor:
        """act: [d_ff] SwiGLU activations (act_fn(gate(x)) * up(x)) -> bool [d_ff]."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class TopMassPolicy(MaskPolicy):
    """Keep the smallest neuron set carrying `mass` fraction of total |activation| mass.

    training-free, per-query adaptive: easy queries naturally keep fewer neurons.
    """

    def __init__(self, mass: float = 0.9):
        if not 0.0 < mass <= 1.0:
            raise ValueError(f"mass must be in (0, 1], got {mass}")
        self.mass = mass

    @property
    def name(self) -> str:
        return f"top{int(self.mass * 100)}mass"

    def mask(self, act: torch.Tensor) -> torch.Tensor:
        flat = act.detach().abs().float().flatten()
        d_ff = flat.numel()
        if self.mass >= 1.0:
            return torch.ones(d_ff, dtype=torch.bool, device=act.device)
        k = max(1, int(d_ff * 0.02))  # never below 2% of neurons
        sorted_idx = torch.argsort(flat, descending=True)
        cum = torch.cumsum(flat[sorted_idx], dim=0)
        cutoff = int((cum / cum[-1] < self.mass).sum().item()) + 1
        keep = torch.zeros(d_ff, dtype=torch.bool, device=act.device)
        keep[sorted_idx[: max(k, cutoff)]] = True
        return keep


class ThresholdPolicy(MaskPolicy):
    """Keep neurons with |act| > frac * max|act| (Deja Vu style threshold)."""

    def __init__(self, frac: float = 0.01):
        self.frac = frac

    @property
    def name(self) -> str:
        return f"thresh{self.frac:g}"

    def mask(self, act: torch.Tensor) -> torch.Tensor:
        flat = act.detach().abs().float().flatten()
        keep = flat > flat.max() * self.frac
        if keep.sum() == 0:
            keep[flat.argmax()] = True
        return keep
