"""Resource monitoring for experiments: RAM, CPU, GPU, power sampling in a background thread."""

import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class ResourceReport:
    """Aggregated resource usage over a monitored window."""

    duration_s: float = 0.0
    rss_mb_avg: float = 0.0
    rss_mb_peak: float = 0.0
    sys_ram_used_gb_avg: float = 0.0
    cpu_pct_avg: float = 0.0
    cpu_pct_peak: float = 0.0
    battery_w_avg: float = 0.0  # discharge rate in W (0 if on AC / unknown)
    battery_pct_start: float = 0.0
    battery_pct_end: float = 0.0
    samples: int = 0

    def summary(self) -> str:
        return (
            f"RSS avg {self.rss_mb_avg:.0f} MB (peak {self.rss_mb_peak:.0f} MB) | "
            f"sys RAM {self.sys_ram_used_gb_avg:.1f} GB | "
            f"CPU {self.cpu_pct_avg:.0f}% (peak {self.cpu_pct_peak:.0f}%) | "
            f"battery {self.battery_w_avg:.1f} W "
            f"({self.battery_pct_start:.0f}%->{self.battery_pct_end:.0f}%)"
        )


class ResourceMonitor:
    """Sample process/system resources at a fixed interval on a daemon thread.

    Battery discharge rate is read from the root/wmi BatteryStatus class via a
    lightweight PowerShell one-shot per sample batch (skip silently on failure,
    e.g. desktop machines or when AC power makes the field meaningless).
    """

    def __init__(self, interval_s: float = 0.5, battery_every_n: int = 10):
        self.interval_s = interval_s
        self.battery_every_n = battery_every_n
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss: list[float] = []
        self._sys: list[float] = []
        self._cpu: list[float] = []
        self._batt_w: list[float] = []
        self._t0 = 0.0

    # -- battery -----------------------------------------------------------
    @staticmethod
    def _read_battery() -> tuple[float, float]:
        """Return (discharge_watts, percent). watts=0 when on AC or unknown."""
        pct = 0.0
        try:
            b = psutil.sensors_battery()
            if b is not None:
                pct = float(b.percent)
        except Exception:
            pass
        watts = 0.0
        try:
            import subprocess

            out = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus "
                    "-ErrorAction Stop | Measure-Object -Property DischargeRate -Sum).Sum",
                ],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            mw = float(out) if out and out.lower() != "0" else 0.0
            watts = mw / 1000.0
        except Exception:
            pass
        return watts, pct

    # -- lifecycle -----------------------------------------------------------
    def _loop(self) -> None:
        i = 0
        pct0 = 0.0
        pct = 0.0
        while not self._stop.is_set():
            try:
                self._rss.append(self._proc.memory_info().rss / 1048576)
                self._sys.append(psutil.virtual_memory().used / 1073741824)
                self._cpu.append(psutil.cpu_percent(interval=None))
                if i % self.battery_every_n == 0:
                    w, pct = self._read_battery()
                    if w > 0:
                        self._batt_w.append(w)
                    if i == 0:
                        pct0 = pct
                i += 1
            except Exception:
                pass
            self._stop.wait(self.interval_s)
        self._pct_start, self._pct_end = pct0, pct

    def __enter__(self) -> "ResourceMonitor":
        psutil.cpu_percent(interval=None)  # prime
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return None

    def report(self) -> ResourceReport:
        def avg(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        return ResourceReport(
            duration_s=time.perf_counter() - self._t0,
            rss_mb_avg=avg(self._rss),
            rss_mb_peak=max(self._rss) if self._rss else 0.0,
            sys_ram_used_gb_avg=avg(self._sys),
            cpu_pct_avg=avg(self._cpu),
            cpu_pct_peak=max(self._cpu) if self._cpu else 0.0,
            battery_w_avg=avg(self._batt_w),
            battery_pct_start=getattr(self, "_pct_start", 0.0),
            battery_pct_end=getattr(self, "_pct_end", 0.0),
            samples=len(self._rss),
        )
