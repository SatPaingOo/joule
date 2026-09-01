"""Resource Governor — auto + manual control of RAM / CPU threads / precision.

The control plane for "any model on any device": detects the machine
(RAM free, cores, NPU presence), lets the user override every knob, and
exposes a status report. Backends pick from this governor's decisions.

Usage (serve):
  --budget-gb 8          manual RAM cap for the expert pool
  --auto-budget          cap at 40% of currently free RAM
  --threads 8            CPU worker threads
  --precision q4|bf16    expert store tier
  --backend auto|native  engine backend (native = fused C kernel)
  --profile battery|balanced|performance   one-shot preset
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import psutil


@dataclass
class DeviceProfile:
    total_ram_gb: float
    free_ram_gb: float
    cores: int
    logical_cores: int
    npu: bool = False
    gpu: str = ""
    battery: bool = False

    @property
    def suggested_budget_gb(self) -> float:
        # enough for the Q4 working set + safety margin, capped by free RAM
        return max(1.0, round(min(self.free_ram_gb * 0.4, 12.0), 1))


def detect_device() -> DeviceProfile:
    vm = psutil.virtual_memory()
    cores = psutil.cpu_count(logical=False) or 4
    logi = psutil.cpu_count() or cores
    p = DeviceProfile(
        total_ram_gb=vm.total / 1073741824,
        free_ram_gb=vm.available / 1073741824,
        cores=cores,
        logical_cores=logi,
    )
    # NPU / GPU hints (best-effort, no hard dependency)
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, timeout=10).stdout.lower()
        p.gpu = out.strip().splitlines()[0] if out.strip() else ""
        p.npu = "ryzen" in out and "radeon" in out  # XDNA NPU present on Ryzen AI
    except Exception:
        pass
    try:
        b = psutil.sensors_battery()
        p.battery = b is not None and b.power_plugged is False
    except Exception:
        pass
    return p


@dataclass
class GovernorConfig:
    budget_gb: float = 8.0
    auto_budget: bool = False
    threads: int = 8
    precision: str = "q4"          # q4 | bf16
    backend: str = "auto"          # auto | native | pool
    profile: str = "balanced"      # battery | balanced | performance

    def resolve(self, dev: DeviceProfile) -> "GovernorConfig":
        c = GovernorConfig(**self.__dict__)
        # profile presets
        if self.profile == "battery":
            c.threads = max(2, dev.cores // 2)
            c.precision = "q4"
            c.budget_gb = min(c.budget_gb, dev.suggested_budget_gb)
        elif self.profile == "performance":
            c.threads = dev.cores
            c.budget_gb = max(c.budget_gb, 10.0)
        else:  # balanced
            c.threads = min(self.threads, dev.cores)
        if self.auto_budget:
            c.budget_gb = dev.suggested_budget_gb
        # backend auto: prefer the control-plane decision (device-adaptive)
        if c.backend == "auto":
            try:
                from jouleai.control import detect_device, AutoSelector, ModelInfo
                # device-adaptive: native only when the model fits RAM with margin
                dev2 = detect_device()
                # heuristic stays device-relative, not a fixed 24GB constant
                est = max(4.0, dev2.total_ram_gb * 0.5)
                c.backend = "native" if dev2.free_ram_gb >= est else "pool"
            except Exception:
                c.backend = "pool"
        return c

    def status(self) -> dict:
        return {
            "budget_gb": self.budget_gb,
            "auto_budget": self.auto_budget,
            "threads": self.threads,
            "precision": self.precision,
            "backend": self.backend,
            "profile": self.profile,
        }


def governor_from_args(args) -> tuple[GovernorConfig, DeviceProfile]:
    dev = detect_device()
    cfg = GovernorConfig(
        budget_gb=getattr(args, "budget_gb", 8.0),
        auto_budget=getattr(args, "auto_budget", False),
        threads=getattr(args, "threads", 8),
        precision=getattr(args, "precision", "q4"),
        backend=getattr(args, "backend", "auto"),
        profile=getattr(args, "profile", "balanced"),
    )
    cfg = cfg.resolve(dev)
    return cfg, dev
