"""Joule package configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JouleConfig:
    """All configuration in one place."""
    model_path: str                    # path to model store (weight files)
    cache_path: str | None = None      # answer cache persistence path
    tau_cache: float = 0.85            # cache hit threshold
    tau_local: float = 0.60            # local model threshold
    probe_depth: int = 8               # layers to probe before decision
    confidence_threshold: float = 0.90 # early exit confidence floor
    max_new_tokens: int = 64
    dtype: str = "auto"                # "auto" | "bfloat16" | "float16" | "float32"
    device: str = "auto"               # "auto" | "cpu" | "cuda"
    cache_capacity: int = 100_000      # max cache entries
