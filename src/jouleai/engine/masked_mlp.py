"""Masked-MLP patching: apply per-layer neuron masks inside a Qwen2-style model.

Two hook modes:
  capture — record SwiGLU activations per layer (mask=None pass-through)
  apply   — zero out non-kept neurons before down_proj (mask given)

Loading accounting: bytes actually addressed per layer =
    keep_n * (2 * d_model + d_model) * dtype_bytes   (gate row + up row + down col)
vs full d_ff * 3 * d_model * dtype_bytes.
"""

from __future__ import annotations

import torch
from torch import nn


class MaskedMLPController:
    """Install forward hooks on every decoder layer's MLP of a Qwen2-style model."""

    def __init__(self, model, hook_point: str = "model.layers"):
        self.model = model
        self.layers = model.get_submodule(hook_point)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self.mode: str | None = None
        self.masks: list[torch.Tensor | None] = [None] * len(self.layers)
        self.captured: list[torch.Tensor | None] = [None] * len(self.layers)
        self._mk_handles()

    def _mk_handles(self) -> None:
        for i, layer in enumerate(self.layers):
            mlp = layer.mlp

            def make_pre(idx: int):
                def pre(_mod: nn.Module, args: tuple) -> tuple:
                    # down_proj input == act_fn(gate(x)) * up(x), shape [B, T, d_ff]
                    if self.mode == "capture":
                        self.captured[idx] = args[0][-1, -1, :].detach().clone()
                    elif self.mode == "apply" and args[0].shape[1] == 1:
                        # decode steps only (T==1): the prompt prefill must stay
                        # exact or context encoding collapses (measured: garbage).
                        m = self.masks[idx]
                        if m is not None:
                            return (args[0] * m.to(args[0].dtype).view(1, 1, -1),)
                    return args

                return pre

            self._hooks.append(mlp.down_proj.register_forward_pre_hook(make_pre(i)))

    # -- public API ----------------------------------------------------------
    def start_capture(self) -> None:
        self.mode, self.captured = "capture", [None] * len(self.layers)

    def start_apply(self, masks: list[torch.Tensor]) -> None:
        self.mode = "apply"
        self.masks = [m.to(self.model.device) for m in masks]

    def stop(self) -> None:
        self.mode, self.masks = None, [None] * len(self.layers)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -- accounting ----------------------------------------------------------
    def loading_report(self, keep_ratios: list[float] | None = None) -> dict:
        """FFN bytes fraction kept, per the gate/up/down row accounting."""
        ratios = keep_ratios
        if ratios is None:
            ratios = [
                (float((m.sum() / m.numel()).item()) if m is not None else 1.0)
                for m in self.masks
            ]
        return {
            "per_layer_keep": ratios,
            "ffn_fraction_loaded": sum(ratios) / len(ratios),
        }
