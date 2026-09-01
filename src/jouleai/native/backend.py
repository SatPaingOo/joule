"""KernelBackend — OOP abstraction over the native compute kernels.

SOLID:
- S: one backend = one ISA/strategy (scalar/avx2/avx512), one file per layer
- O: new ISA = new KernelBackend subclass, no changes to callers
- L: every backend satisfies the same interface (substitutable)
- I: backends expose only what the engine needs (q4_gemv, q4_gemm, decode)
- D: the engine depends on KernelBackend, not on any concrete DLL

Usage:
    backend = KernelBackend.auto()          # picks best available ISA
    backend.q4_gemv(x, packed, scales, m, d)
    backend.q4_gemm(out, X, packed, scales, m, d)
"""

from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

_NATIVE_DIR = Path(__file__).parent

# DLL cache: one handle per kernel, shared across all consumers (SOLID D —
# the engine depends on this module, not on ctypes.CDLL paths).
_DLL_CACHE: dict[str, ctypes.CDLL] = {}


def get_dll(name: str) -> ctypes.CDLL:
    """Load (and cache) a native kernel DLL by name (no '.dll' suffix).

    The single access point for all kernel DLLs — decoders, backends, and
    tests share one handle per kernel instead of each calling
    ctypes.CDLL("...") with a hardcoded path. Adding a new kernel = one
    name here, no caller changes (open/closed).
    """
    if name not in _DLL_CACHE:
        _DLL_CACHE[name] = ctypes.CDLL(str(_NATIVE_DIR / f"{name}.dll"))
    return _DLL_CACHE[name]


class KernelBackend(ABC):
    """Interface every native kernel backend implements."""

    name: str = "base"

    @abstractmethod
    def q4_gemv(self, x_f32: torch.Tensor, packed: np.ndarray,
                scales: np.ndarray, m: int, d: int, group: int = 64) -> torch.Tensor:
        """out[m] = dequant_q4(packed, scales) @ x[d] (no dequant materialise)."""

    @abstractmethod
    def q4_gemm(self, out_f32: torch.Tensor, X_f32: torch.Tensor,
                packed: np.ndarray, scales: np.ndarray,
                m: int, d: int, group: int = 64) -> None:
        """out[T,m] (in place) = X[T,d] @ dequant(W)^T."""

    @staticmethod
    def auto() -> "KernelBackend":
        """Best available backend for this machine."""
        for impl in (AVX2Backend, ScalarBackend):
            try:
                return impl()
            except Exception:
                continue
        raise RuntimeError("no kernel backend available")


class ScalarBackend(KernelBackend):
    """Scalar reference kernels (quant_gemv.dll)."""

    name = "scalar"

    def __init__(self):
        self._dll = get_dll("quant_gemv")
        self._dll.q4_gemv_f32.restype = None

    def q4_gemv(self, x_f32, packed, scales, m, d, group=64):
        assert x_f32.dtype == torch.float32 and x_f32.numel() == d
        out = torch.empty(m, dtype=torch.float32)
        self._dll.q4_gemv_f32(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(x_f32.data_ptr()),
            ctypes.c_void_p(packed.ctypes.data),
            ctypes.c_void_p(scales.ctypes.data),
            m, d, group)
        return out

    def q4_gemm(self, out_f32, X_f32, packed, scales, m, d, group=64):
        raise NotImplementedError("scalar GEMM: use expert_ffn backend")


class AVX2Backend(KernelBackend):
    """AVX2 fused kernels (expert_ffn.dll: q4_gemm_f32)."""

    name = "avx2"

    def __init__(self):
        self._dll = get_dll("expert_ffn")
        self._dll.q4_gemm_f32.restype = None

    def q4_gemv(self, x_f32, packed, scales, m, d, group=64):
        raise NotImplementedError("GEMV: use q4_gemm with T=1")

    def q4_gemm(self, out_f32, X_f32, packed, scales, m, d, group=64):
        T = X_f32.shape[0]
        self._dll.q4_gemm_f32(
            ctypes.c_void_p(out_f32.data_ptr()),
            ctypes.c_void_p(X_f32.data_ptr()),
            T,
            ctypes.c_void_p(packed.ctypes.data),
            ctypes.c_void_p(scales.ctypes.data),
            m, d, group)


class DecodeBackend(KernelBackend):
    """Full 48-layer decode kernel (decode_kernel.dll) — the fastest path."""

    name = "decode"

    def __init__(self):
        self._dll = get_dll("decode_kernel")
        # full-decode API is exposed via NativeDecoder (decoder3.py)
        from jouleai.native.decoder3 import NativeDecoder
        self.decoder_cls = NativeDecoder

    def q4_gemv(self, x_f32, packed, scales, m, d, group=64):
        raise NotImplementedError("decode backend: use NativeDecoder")

    def q4_gemm(self, out_f32, X_f32, packed, scales, m, d, group=64):
        raise NotImplementedError("decode backend: use NativeDecoder")
