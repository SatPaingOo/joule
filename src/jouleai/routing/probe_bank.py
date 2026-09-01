"""ProbeBank — trained per-layer linear probes that predict FFN activations.

The Sense Layer Converter's predictor: during a one-time calibration pass the
engine collects (hidden, activation) pairs per layer; closed-form ridge
regression learns W_l: act_l ≈ h_l @ W_l. At decode time the probe predicts
activations directly from the hidden state — no gate/up scan needed — and the
mask policy thresholds the prediction into neuron keep-sets.

Linear + closed-form: no SGD, CPU-friendly, minutes for 7B.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from jouleai.routing.mask_policy import MaskPolicy


class ProbeBank:
    """Per-layer probe weights [d, d_ff]; act_pred = h @ W."""

    def __init__(self, weights: dict[str, torch.Tensor]):
        self.layers: list[torch.Tensor | None] = []
        i = 0
        while f"model.layers.{i}.probe.weight" in weights:
            self.layers.append(weights[f"model.layers.{i}.probe.weight"])
            i += 1
        if not self.layers:
            raise ValueError("no probe tensors found in weights")

    @classmethod
    def load(cls, path: str | Path) -> "ProbeBank":
        return cls(load_file(str(path)))

    def save(self, path: str | Path) -> None:
        out = {f"model.layers.{i}.probe.weight": w.contiguous()
               for i, w in enumerate(self.layers)}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(out, str(path))

    def __len__(self) -> int:
        return len(self.layers)

    def predict_act(self, l: int, h: torch.Tensor) -> torch.Tensor:
        """h [.., d] -> predicted activations [.., d_ff] (bf16 compute)."""
        return h @ self.layers[l]

    def mask_for(self, l: int, h: torch.Tensor, policy: MaskPolicy) -> torch.Tensor:
        """Predicted keep-mask for the last position of h [1, T, d]."""
        act = self.predict_act(l, h)[0, -1]
        return policy.mask(act.abs())


def train_probes(x: torch.Tensor, y: torch.Tensor,
                 lam_frac: float = 0.1) -> torch.Tensor:
    """Closed-form ridge: W = (XᵀX + λI)⁻¹ XᵀY.

    x: [N, d] hidden states, y: [N, d_ff] activations (float32, N can be < d —
    ridge handles the underdetermined case; lam_frac scales λ by mean diagonal).
    """
    a = x.T @ x
    a += lam_frac * a.diagonal().mean() * torch.eye(a.shape[0], dtype=a.dtype)
    b = x.T @ y
    return torch.linalg.solve(a, b)  # [d, d_ff]
