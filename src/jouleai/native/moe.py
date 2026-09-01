"""Native full-FFN decode: one C expert_job per expert, threads via kernel32.

Python builds k ExpertJob structs pointing at raw Q4 records + x, spawns k
threads (ctypes releases the GIL, so they run truly parallel), waits, and
combines the weighted outputs in one torch op. Per layer: 8 thread spawns +
1 combine — Python orchestration collapses from ~24 dispatches to this.
"""

from __future__ import annotations

import ctypes

import numpy as np
import torch

from jouleai.native.backend import get_dll

_DLL = get_dll("expert_ffn")
THREADFUNC = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
_fn_thread_entry = THREADFUNC(_DLL.thread_entry)  # keep reference alive
_k32 = ctypes.WinDLL("kernel32")
_k32.CreateThread.restype = ctypes.c_void_p
_k32.CreateThread.argtypes = [ctypes.c_void_p, ctypes.c_size_t, THREADFUNC,
                              ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
_k32.WaitForMultipleObjects.restype = ctypes.c_uint32
_k32.WaitForMultipleObjects.argtypes = [ctypes.c_uint32,
                                        ctypes.POINTER(ctypes.c_void_p),
                                        ctypes.c_int, ctypes.c_uint32]
INFINITE = 0xFFFFFFFF


class ExpertJob(ctypes.Structure):
    _fields_ = [
        ("pk_g", ctypes.c_void_p), ("sc_g", ctypes.c_void_p),
        ("pk_u", ctypes.c_void_p), ("sc_u", ctypes.c_void_p),
        ("pk_d", ctypes.c_void_p), ("sc_d", ctypes.c_void_p),
        ("m_gg", ctypes.c_int), ("d_in", ctypes.c_int),
        ("m_d", ctypes.c_int), ("d_m", ctypes.c_int),
        ("group", ctypes.c_int),
        ("x", ctypes.c_void_p), ("scratch", ctypes.c_void_p),
        ("prob", ctypes.c_float),
    ]


class NativeMoE:
    """Reusable state: thread funcs, scratch buffers, job structs."""

    def __init__(self, d_in: int, m_gg: int, k: int, group: int = 64):
        self.d_in, self.m_gg, self.k, self.group = d_in, m_gg, k, group
        self.jobs = [ExpertJob() for _ in range(k)]
        self.job_ptrs = (ctypes.c_void_p * k)(
            *[ctypes.cast(ctypes.byref(j), ctypes.c_void_p) for j in self.jobs])
        self.handles = (ctypes.c_void_p * k)()
        self.scratch: list[torch.Tensor] = []

    def ffn(self, eng, l: int, h: torch.Tensor, pool,
            experts: list[int], w: torch.Tensor,
            executor=None) -> torch.Tensor:
        """out [1,1,d_in] bf16 — full FFN for one decode token.

        One expert_job C call per expert (3 fused matvecs + silu + weighting),
        dispatched on the shared ThreadPoolExecutor (GIL released inside
        ctypes — true parallelism without win32 thread plumbing).
        """
        x = h[0, -1].float()
        recs = [pool.ensure(l, e) for e in experts]
        k = len(experts)
        scratch = [torch.empty(self.m_gg + self.d_in, dtype=torch.float32)
                   for _ in range(k)]
        self.scratch = scratch  # keep alive
        for i in range(k):
            j = self.jobs[i]
            sc_g, pk_g, n_g = recs[i][0]
            sc_u, pk_u, n_u = recs[i][1]
            sc_d, pk_d, n_d = recs[i][2]
            j.pk_g = pk_g.ctypes.data
            j.sc_g = sc_g.ctypes.data
            j.pk_u = pk_u.ctypes.data
            j.sc_u = sc_u.ctypes.data
            j.pk_d = pk_d.ctypes.data
            j.sc_d = sc_d.ctypes.data
            j.m_gg = self.m_gg
            j.d_in = self.d_in
            j.m_d = self.d_in
            j.d_m = self.m_gg
            j.group = self.group
            j.x = x.data_ptr()
            j.scratch = scratch[i].data_ptr()
            j.prob = float(w[0, i])

        def run(i: int):
            _DLL.expert_job(ctypes.byref(self.jobs[i]))

        if executor is not None:
            list(executor.map(run, range(k)))
        else:
            for i in range(k):
                run(i)
        out = torch.zeros(self.d_in, dtype=torch.float32)
        for i in range(k):
            out += scratch[i][self.m_gg:]
        return out.unsqueeze(0).unsqueeze(0).to(h.dtype)
