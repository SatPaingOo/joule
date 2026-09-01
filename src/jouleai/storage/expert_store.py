"""Expert-level store + LRU pool — the storage layer for MoE-scale serving.

ExpertStore: partitions a model's FFN neurons into E contiguous experts per
layer on top of SenseWeightStore + the transposed down_proj store, so an
expert is a contiguous neuron-row range (fast gather).

ExpertLRUPool: a RAM-budgeted cache of experts with router-guided fill,
LRU eviction, and IO accounting — "load only what you need, release after".
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from jouleai.storage.weight_store import SenseWeightStore


@dataclass
class PoolStats:
    hits: int = 0
    misses: int = 0
    io_bytes: int = 0
    evictions: int = 0
    peak_resident: int = 0  # bytes

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


class ExpertStore:
    """Address experts as contiguous neuron-row ranges per layer."""

    def __init__(self, store: SenseWeightStore, down_t: SenseWeightStore,
                 n_layers: int, d_ff: int, n_experts: int):
        if d_ff % n_experts != 0:
            raise ValueError(f"d_ff {d_ff} not divisible by n_experts {n_experts}")
        self.store, self.down_t = store, down_t
        self.n_layers, self.d_ff, self.n_experts = n_layers, d_ff, n_experts
        self.expert_neurons = d_ff // n_experts
        # bytes per expert: gate row + up row + down row, bf16
        self.expert_bytes: list[int] = []
        self.d_model = store.shape_of("model.embed_tokens.weight")[1]
        for l in range(n_layers):
            self.expert_bytes.append(self.expert_neurons * 3 * self.d_model * 2)

    def neuron_range(self, l: int, e: int) -> tuple[int, int]:
        s = e * self.expert_neurons
        return s, s + self.expert_neurons

    def expert_size_bytes(self, l: int, e: int) -> int:
        return self.expert_neurons * 3 * self.d_model * 2

    def gather(self, l: int, e: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (gate, up, down) rows for expert e of layer l, each [m, d]."""
        p = f"model.layers.{l}.mlp"
        lo, hi = self.neuron_range(l, e)
        idx = torch.arange(lo, hi)
        g = self.store.rows(f"{p}.gate_proj.weight", idx).to(torch.bfloat16)
        u = self.store.rows(f"{p}.up_proj.weight", idx).to(torch.bfloat16)
        dn = self.down_t.rows(f"{p}.down_proj_t.weight", idx).to(torch.bfloat16)
        return g, u, dn


class ExpertLRUPool:
    """RAM-budgeted resident set of experts, LRU-evicted, usage-counted."""

    def __init__(self, expert_store: ExpertStore, budget_bytes: int):
        self.es = expert_store
        self.budget = budget_bytes
        self._map: OrderedDict[tuple[int, int], tuple[torch.Tensor, ...]] = OrderedDict()
        self._bytes = 0
        self._use: dict[tuple[int, int], int] = {}
        self.stats = PoolStats()

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._map

    def resident_experts(self) -> int:
        return len(self._map)

    def resident_bytes(self) -> int:
        return self._bytes

    def ensure(self, l: int, e: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return resident expert tensors, gathering from disk on a miss."""
        key = (l, e)
        self._use[key] = self._use.get(key, 0) + 1
        if key in self._map:
            self._map.move_to_end(key)
            self.stats.hits += 1
            return self._map[key]
        self.stats.misses += 1
        eb = self.expert_size_bytes(l, e)
        self.stats.io_bytes += eb
        tensors = self.es.gather(l, e)
        self._bytes += eb
        self._map[key] = tensors
        self._evict_to_budget()
        self.stats.peak_resident = max(self.stats.peak_resident, self._bytes)
        return tensors

    def expert_size_bytes(self, l: int, e: int) -> int:
        return self.es.expert_size_bytes(l, e)

    def prune_to_budget(self, target_bytes: int | None = None) -> int:
        """Evict lowest-usage experts until within budget (call after prefill:
        keeps the experts decode is most likely to reuse). Returns evicted n."""
        target = self.budget if target_bytes is None else target_bytes
        if self._bytes <= target:
            return 0
        ranked = sorted(self._map, key=lambda k: self._use.get(k, 0))
        evicted = 0
        for key in ranked:
            if self._bytes <= target:
                break
            tensors = self._map.pop(key)
            self._bytes -= self.expert_size_bytes(*key)
            self._use.pop(key, None)
            del tensors
            evicted += 1
            self.stats.evictions += 1
        return evicted

    def _evict_to_budget(self) -> None:
        while self._bytes > self.budget and len(self._map) > 1:
            k, _ = self._map.popitem(last=False)  # evict LRU
            self._bytes -= self.expert_size_bytes(*k)
            self._use.pop(k, None)
            self.stats.evictions += 1

    def clear(self) -> None:
        self._map.clear()
        self._bytes = 0


class MoERouter:
    """Synthetic top-k router (machinery validation).

    Deterministic per-expert unit vectors c_e; score = |h · c_e|. A real MoE
    model replaces this with its trained gate — the plumbing is identical.
    """

    def __init__(self, n_layers: int, d_model: int, n_experts: int,
                 top_k: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.vectors = torch.randn(n_layers, n_experts, d_model, generator=g)
        self.vectors /= self.vectors.norm(dim=-1, keepdim=True)
        self.top_k = top_k

    def select(self, l: int, h: torch.Tensor) -> torch.Tensor:
        """h [1, T, d] (last token used) -> sorted expert indices [k]."""
        v = self.vectors[l].to(h.dtype)                       # [E, d]
        scores = (h[0, -1].float() @ v.float().T).abs()       # [E]
        k = min(self.top_k, scores.numel())
        return torch.topk(scores, k).indices.sort().values


class Bf16ExpertPool:
    """Expert pool over the ORIGINAL bf16 safetensors (no quantisation).

    The quality tier for machines with RAM to spare: reads full tensors,
    zero dequantisation cost, full bf16 precision.
    """

    def __init__(self, store: SenseWeightStore, n_layers: int, n_experts: int,
                 budget_bytes: int):
        self.store = store
        self._inner = ExpertLRUPool(self, budget_bytes)
        self._sizes = [
            [sum(store.bytes_of(f"model.layers.{l}.mlp.experts.{e}.{p}_proj.weight")
                 for p in ("gate", "up", "down")) for e in range(n_experts)]
            for l in range(n_layers)
        ]

    def expert_size_bytes(self, l: int, e: int) -> int:
        return self._sizes[l][e]

    def gather(self, l: int, e: int):
        p = f"model.layers.{l}.mlp.experts.{e}"
        return tuple(self.store.full(f"{p}.{x}_proj.weight")
                     for x in ("gate", "up", "down"))

    def __getattr__(self, name):
        return getattr(self._inner, name)
