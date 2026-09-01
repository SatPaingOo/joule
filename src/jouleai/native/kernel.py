"""ctypes wrapper for quant_gemv.dll (fused Q4-dequant GEMV)."""

from __future__ import annotations

import ctypes

import numpy as np
import torch

from jouleai.native.backend import get_dll

_dll = None


def dll():
    global _dll
    if _dll is None:
        _dll = get_dll("quant_gemv")
        _dll.q4_gemv_f32.restype = None
        # args passed as c_void_p: (out, x, packed, scales, m, d, group)
    return _dll


def q4_gemv(x_f32: torch.Tensor, packed: np.ndarray, scales: np.ndarray,
            m: int, d: int, group: int = 64) -> torch.Tensor:
    """out[m] fp32 = dequant_q4(packed, scales) @ x. No dequant materialisation.

    packed/scales are numpy views (may back a memmap — read directly in C).
    x_f32: torch fp32 [d]. Returns torch fp32 [m]. GIL released during call.
    """
    assert x_f32.dtype == torch.float32 and x_f32.numel() == d
    assert packed.dtype == np.uint8 and packed.size == m * d // 2
    out = torch.empty(m, dtype=torch.float32)
    fn = dll().q4_gemv_f32
    fn(ctypes.c_void_p(out.data_ptr()),
       ctypes.c_void_p(x_f32.data_ptr()),
       ctypes.c_void_p(packed.ctypes.data),
       ctypes.c_void_p(scales.ctypes.data),
       m, d, group)
    return out


_DLL2 = get_dll("expert_ffn")
_DLL2.q4_gemm_f32.restype = None


def q4_gemm(out_f32: torch.Tensor, X_f32: torch.Tensor,
            packed: np.ndarray, scales: np.ndarray,
            m: int, d: int, group: int = 64) -> None:
    """out_f32 [T, m] fp32 (modified in place) = X_f32 [T, d] @ dequant(W)^T.

    Fused dequant — the packed expert is never materialised.
    """
    T = X_f32.shape[0]
    _DLL2.q4_gemm_f32(
        ctypes.c_void_p(out_f32.data_ptr()),
        ctypes.c_void_p(X_f32.data_ptr()),
        T,
        ctypes.c_void_p(packed.ctypes.data),
        ctypes.c_void_p(scales.ctypes.data),
        m, d, group,
    )
