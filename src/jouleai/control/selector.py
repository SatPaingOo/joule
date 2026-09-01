"""Auto-select — the adaptive control logic.

Turns (DeviceProfile + model metadata + user overrides) into an ExecutionPlan:
the single set of knobs the engine runs with. This is the "C" control plane:
same model, any device -> the plan adapts (quant tier, backend, batch size,
spec decoding, sparsity, RAM budget, threads).

Everything is data-driven from the profile and the model's own config — no
hardcoded model dimensions. The user can override any knob; auto picks safe
defaults. New heuristics = a new Selector (Strategy), no caller changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from jouleai.control.device import DeviceProfile


@dataclass
class ModelInfo:
    """Model metadata from config.json (what the plan adapts to)."""
    params_b: float = 0.0          # total params (billions)
    active_b: float = 0.0          # active params per token (MoE) or total (dense)
    moe: bool = False
    n_layers: int = 0
    hidden: int = 0
    n_experts: int = 0
    topk: int = 0

    @staticmethod
    def from_config(c: dict) -> "ModelInfo":
        """Estimate param counts from a HF config.json (no weight scan).

        Per layer (standard transformer):
          attention: 4 x d x d (q/k/v/o, with GQA ~2d^2)
          FFN:       3 x d x intermediate  (gate/up/down) — or MoE experts
        MoE total = layers x (attn + E x 3 x d x inter); active uses topk.
        """
        d = c.get("hidden_size", 0)
        L = c.get("num_hidden_layers", 0)
        inter = c.get("moe_intermediate_size") or c.get("intermediate_size", 0)
        E = c.get("num_experts", 0)
        topk = c.get("num_experts_per_tok", 0)
        n_kv = c.get("num_key_value_heads", 1)
        n_h = c.get("num_attention_heads", 1)
        # attention params per layer: q (d*d) + k (d*n_kv*hd) + v (d*n_kv*hd) + o (d*d)
        hd = c.get("head_dim", d // n_h) if d and n_h else 0
        attn = 2 * d * d + 2 * d * n_kv * hd
        if E:
            ffn_total = E * 3 * d * inter
            ffn_active = topk * 3 * d * inter
            total = L * (attn + ffn_total) / 1e9
            active = L * (attn + ffn_active) / 1e9
            return ModelInfo(total, active, True, L, d, E, topk)
        ffn = 3 * d * inter
        total = L * (attn + ffn) / 1e9
        return ModelInfo(total, total, False, L, d, 0, 0)

    @property
    def q4_gb(self) -> float:
        """Rough Q4 size: ~0.55 bytes/param (4-bit + scales/overhead)."""
        return round(self.params_b * 0.55, 1)

    @property
    def bf16_gb(self) -> float:
        return round(self.params_b * 2.0, 1)

    @property
    def active_q4_gb(self) -> float:
        """Active set per token (Q4) — the per-token read size."""
        return round(self.active_b * 0.55, 2)


@dataclass
class ExecutionPlan:
    budget_gb: float = 4.0
    threads: int = 4
    precision: str = "q4"          # q4 | bf16 | q8
    backend: str = "pool"          # native | pool
    batch: int = 1                 # decode batch size (aggregate throughput)
    spec: bool = False             # speculative decoding on/off
    sparsity: bool = False         # neuron sparsity (verify-gated) on/off
    max_concurrent: int = 4
    device_tier: str = "low"
    rationale: list[str] = field(default_factory=list)

    def status(self) -> dict:
        return {
            "budget_gb": self.budget_gb, "threads": self.threads,
            "precision": self.precision, "backend": self.backend,
            "batch": self.batch, "spec": self.spec, "sparsity": self.sparsity,
            "max_concurrent": self.max_concurrent, "tier": self.device_tier,
            "rationale": self.rationale,
        }


class Selector(ABC):
    """Auto-select strategy. Implementations turn a profile into a plan."""

    @abstractmethod
    def select(self, dev: DeviceProfile, model: ModelInfo,
               overrides: dict | None = None) -> ExecutionPlan: ...


class AutoSelector(Selector):
    """Data-driven default selector (the main auto strategy)."""

    def select(self, dev: DeviceProfile, model: ModelInfo,
               overrides: dict | None = None) -> ExecutionPlan:
        overrides = overrides or {}
        plan = ExecutionPlan(device_tier=dev.tier)
        r = plan.rationale

        # ---- threads: physical cores (respect user) ----
        plan.threads = dev.cores
        r.append(f"cores={dev.cores}")

        # ---- RAM budget: fit the working set, capped by free RAM ----
        active = model.active_q4_gb
        # budget = active set + safety, but never exceed 40% free RAM
        want = max(active * 1.5, 1.0)
        plan.budget_gb = round(min(want, dev.safe_budget_gb), 1)
        r.append(f"active={active}GB -> budget={plan.budget_gb}GB "
                 f"(free={dev.free_ram_gb:.0f}GB)")

        # ---- precision / backend by device tier + model size ----
        if model.q4_gb > dev.free_ram_gb * 0.5:
            # model too big to hold resident -> disk-backed pool + Q4
            plan.backend = "pool"
            plan.precision = "q4"
            r.append("model > 50% RAM -> Q4 pool (disk-backed)")
        elif dev.tier == "high" and model.q4_gb <= dev.free_ram_gb * 0.8:
            # plenty of RAM + bandwidth -> native resident kernel
            plan.backend = "native"
            plan.precision = "bf16" if dev.mem_bw_gb_s >= 100 else "q4"
            r.append("high tier -> native backend")
        else:
            plan.backend = "pool"
            plan.precision = "q4"
            r.append("mid/low tier -> Q4 pool")

        # ---- batch: more cores / bandwidth -> bigger decode batch ----
        if dev.tier == "high":
            plan.batch = 8
        elif dev.tier == "mid":
            plan.batch = 4
        else:
            plan.batch = 1
        plan.max_concurrent = plan.batch
        r.append(f"tier={dev.tier} -> batch={plan.batch}")

        # ---- spec decode: enable on bandwidth-bound machines with a draft ----
        # (draft model availability is decided by the caller; flag readiness)
        plan.spec = dev.mem_bw_gb_s >= 40
        r.append(f"spec={'on' if plan.spec else 'off'} (bw={dev.mem_bw_gb_s:.0f}GB/s)")

        # ---- sparsity: off by default (quality risk) unless user opts in ----
        plan.sparsity = bool(overrides.get("sparsity", False))
        if plan.sparsity:
            r.append("sparsity=on (verify-gated, user-opted)")

        # ---- apply user overrides last (single control point) ----
        for k in ("budget_gb", "threads", "precision", "backend", "batch",
                  "spec", "sparsity", "max_concurrent"):
            if k in overrides and overrides[k] is not None:
                setattr(plan, k, overrides[k])
        return plan
