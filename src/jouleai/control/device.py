"""Device detection — cross-OS hardware profile (Windows / macOS / Linux).

Produces a DeviceProfile describing what the machine has: OS, RAM, CPU cores,
memory bandwidth estimate, GPU, NPU, battery. Every detector is a small
strategy; new hardware probes are added as new Detector classes (no changes
to callers). Detection is best-effort — a failed probe degrades to a safe
default, never crashes serve.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass
class DeviceProfile:
    os: str = platform.system().lower()          # "windows" | "darwin" | "linux"
    arch: str = platform.machine().lower() or "x86_64"
    total_ram_gb: float = 8.0
    free_ram_gb: float = 4.0
    cores: int = 4
    logical_cores: int = 8
    mem_bw_gb_s: float = 20.0                    # estimated memory bandwidth
    gpu: str = ""                                # e.g. "Radeon 890M"
    gpu_ram_gb: float = 0.0
    npu: bool = False
    npu_name: str = ""
    battery: bool = False
    is_arm: bool = False

    @property
    def tier(self) -> str:
        """Device capability tier — drives auto-select defaults.

        Weights RAM (free) + bandwidth most; integrated-GPU shared memory is
        a bonus, not the main signal (it shares the same RAM).
        """
        if self.mem_bw_gb_s >= 100 and self.free_ram_gb >= 32:
            return "high"
        if self.mem_bw_gb_s >= 50 or self.free_ram_gb >= 24:
            return "mid"
        return "low"

    @property
    def safe_budget_gb(self) -> float:
        """RAM budget for the working set: fraction of free RAM, capped."""
        return max(1.0, round(min(self.free_ram_gb * 0.4, 16.0), 1))


class DeviceDetector(ABC):
    """A single probe. Implementations fill one aspect of the profile."""

    @abstractmethod
    def apply(self, p: DeviceProfile) -> None: ...


class PsutilDetector(DeviceDetector):
    """RAM / CPU via psutil (all platforms)."""

    def apply(self, p: DeviceProfile) -> None:
        if psutil is None:
            return
        try:
            vm = psutil.virtual_memory()
            p.total_ram_gb = vm.total / 1073741824
            p.free_ram_gb = vm.available / 1073741824
        except Exception:
            pass
        try:
            p.cores = psutil.cpu_count(logical=False) or p.cores
            p.logical_cores = psutil.cpu_count() or p.logical_cores
        except Exception:
            pass
        try:
            b = psutil.sensors_battery()
            p.battery = b is not None and b.power_plugged is False
        except Exception:
            pass


class BandwidthEstimator(DeviceDetector):
    """Memory bandwidth estimate from CPU/OS heuristics.

    DDR4 ~25 GB/s, DDR5 ~50-60, LPDDR5x unified ~100+, Apple M-series ~100-400.
    Used to pick the speed tier (what's achievable on this machine).
    """

    def apply(self, p: DeviceProfile) -> None:
        if p.os == "darwin":
            # Apple Silicon unified memory: bandwidth scales with RAM
            p.mem_bw_gb_s = min(400.0, max(60.0, p.total_ram_gb * 8))
            p.is_arm = p.arch in ("arm64", "aarch64")
        elif p.os == "linux":
            # AMD/Intel desktop CPU + DDR: rough by generation (best-effort)
            p.mem_bw_gb_s = 40.0 if p.total_ram_gb >= 16 else 25.0
            p.is_arm = p.arch in ("arm64", "aarch64", "aarch64")
        else:  # windows
            # Try to read the memory speed via powershell (DDR4=3200 / DDR5=5600)
            try:
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Speed -Average).Average"],
                    capture_output=True, text=True, timeout=8).stdout.strip()
                mhz = float(out) if out else 0.0
                # single channel ~8 bytes/cycle; dual ~16; quad ~32
                p.mem_bw_gb_s = round(min(100.0, mhz * 16 / 1000), 1) if mhz else 35.0
            except Exception:
                p.mem_bw_gb_s = 35.0


class GpuDetector(DeviceDetector):
    """GPU presence via platform tools (best-effort, never fatal)."""

    def apply(self, p: DeviceProfile) -> None:
        try:
            if p.os == "windows":
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                    capture_output=True, text=True, timeout=8).stdout.lower()
                names = [n.strip() for n in out.splitlines() if n.strip()]
                p.gpu = names[0] if names else ""
                g = p.gpu.lower()
                # NPU / AI-accelerator hints (best-effort by GPU name)
                if "ryzen ai" in g or "xdn" in g:
                    p.npu, p.npu_name = True, "AMD XDNA"
                elif "radeon" in g and any(k in g for k in
                        ("860m", "870m", "880m", "890m", "strix", "halo")):
                    p.npu, p.npu_name = True, "AMD RDNA iGPU (NPU-class)"
                # integrated Radeon shares system RAM (unified-ish)
                if "radeon" in g and ("m" in g or "860" in g or "870" in g
                                      or "880" in g or "890" in g):
                    p.gpu_ram_gb = min(p.total_ram_gb * 0.5, 24.0)
            elif p.os == "linux":
                if shutil.which("nvidia-smi"):
                    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                          "--format=csv,noheader"], capture_output=True,
                                         text=True, timeout=8).stdout.strip()
                    if out:
                        p.gpu = out.splitlines()[0]
                        p.gpu_ram_gb = 12.0
            elif p.os == "darwin":
                # Apple Silicon GPU shares unified memory — flag it
                p.gpu = "Apple Silicon" if p.is_arm else ""
                if p.gpu:
                    p.gpu_ram_gb = p.total_ram_gb  # unified memory counts
        except Exception:
            pass


def detect_device() -> DeviceProfile:
    """Run all detectors in order. Safe defaults if anything fails."""
    p = DeviceProfile()
    for d in (PsutilDetector(), BandwidthEstimator(), GpuDetector()):
        try:
            d.apply(p)
        except Exception:
            pass
    return p
