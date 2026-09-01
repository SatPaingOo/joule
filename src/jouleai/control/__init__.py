"""Control plane — device-adaptive execution planning.

The single place that decides HOW a model runs on THIS machine:
  - device.py   → detect OS / RAM / CPU / GPU / NPU / bandwidth
  - selector.py → turn (device + model + overrides) into an ExecutionPlan

Anything that needs a knob (budget, threads, precision, backend, batch,
spec, sparsity) reads it from the plan. New devices/heuristics = new
Detector/Selector, no caller changes (SOLID: open/closed, dependency
inversion).
"""

from jouleai.control.device import DeviceProfile, detect_device
from jouleai.control.selector import AutoSelector, ExecutionPlan, ModelInfo
from jouleai.control.controls import ControlCenter, RuntimeHealth

__all__ = [
    "DeviceProfile", "detect_device",
    "AutoSelector", "ExecutionPlan", "ModelInfo",
    "ControlCenter", "RuntimeHealth",
]


def plan_for(model_info: ModelInfo, overrides: dict | None = None) -> tuple[DeviceProfile, ExecutionPlan]:
    """One-call control: detect device, auto-select the plan."""
    dev = detect_device()
    plan = AutoSelector().select(dev, model_info, overrides)
    return dev, plan
