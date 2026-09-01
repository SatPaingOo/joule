"""ControlCenter — the single control point for the whole system.

Everything the engine needs to know about HOW to run is here:
  - device profile (detected)
  - execution plan (auto-selected + overrides)
  - runtime health (RAM, CPU, pool residency) for adaptive re-tuning

One object, one place to read knobs and one place to adjust them.
The engine never hardcodes a knob — it asks the ControlCenter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jouleai.control.device import DeviceProfile, detect_device
from jouleai.control.selector import AutoSelector, ExecutionPlan, ModelInfo

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass
class RuntimeHealth:
    rss_gb: float = 0.0
    free_ram_gb: float = 0.0
    cpu_pct: float = 0.0
    pool_resident_gb: float = 0.0
    io_gb_per_tok: float = 0.0

    def as_dict(self) -> dict:
        return {
            "rss_gb": round(self.rss_gb, 1),
            "free_ram_gb": round(self.free_ram_gb, 1),
            "cpu_pct": round(self.cpu_pct, 0),
            "pool_resident_gb": round(self.pool_resident_gb, 1),
            "io_gb_per_tok": round(self.io_gb_per_tok, 3),
        }


class ControlCenter:
    """One place: detect device → plan → runtime health → status."""

    def __init__(self, model_info: ModelInfo, overrides: dict | None = None,
                 selector: AutoSelector | None = None):
        self.model = model_info
        self.device = detect_device()
        self.selector = selector or AutoSelector()
        self.plan = self.selector.select(self.device, model_info, overrides)

    # ---- read knobs (engine calls these) ----
    @property
    def budget_gb(self) -> float:
        return self.plan.budget_gb

    @property
    def threads(self) -> int:
        return self.plan.threads

    @property
    def precision(self) -> str:
        return self.plan.precision

    @property
    def backend(self) -> str:
        return self.plan.backend

    @property
    def batch(self) -> int:
        return self.plan.batch

    @property
    def max_concurrent(self) -> int:
        return self.plan.max_concurrent

    # ---- runtime health (updated by the engine each step) ----
    def sample_health(self, pool_resident_gb: float = 0.0,
                      io_gb_per_tok: float = 0.0) -> RuntimeHealth:
        h = RuntimeHealth(pool_resident_gb=pool_resident_gb,
                          io_gb_per_tok=io_gb_per_tok)
        try:
            if psutil is not None:
                h.rss_gb = psutil.Process().memory_info().rss / 1073741824
                h.free_ram_gb = psutil.virtual_memory().available / 1073741824
                h.cpu_pct = psutil.cpu_percent(interval=None)
        except Exception:
            pass
        return h

    # ---- status: the single control-panel view ----
    def status(self) -> dict:
        return {
            "device": {
                "os": self.device.os, "arch": self.device.arch,
                "ram_gb": round(self.device.total_ram_gb, 1),
                "free_gb": round(self.device.free_ram_gb, 1),
                "cores": self.device.cores,
                "logical_cores": self.device.logical_cores,
                "bw_gb_s": round(self.device.mem_bw_gb_s, 0),
                "gpu": self.device.gpu, "npu": self.device.npu_name,
                "tier": self.device.tier,
            },
            "model": {
                "params_b": round(self.model.params_b, 1),
                "active_b": round(self.model.active_b, 1),
                "moe": self.model.moe,
                "q4_gb": self.model.q4_gb,
                "active_q4_gb": self.model.active_q4_gb,
            },
            "plan": self.plan.status(),
            "health": self.sample_health().as_dict(),
        }
