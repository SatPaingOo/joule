"""Batch decode benchmark: decode_layers_batch across B sequences.

Measures aggregate tok/s for B=1..8 (weights read once per B tokens).
Each run: N batch steps, each B sequences decode one token at their own
positions (fresh sequential KV per sequence). Reports wall time and
aggregate tok/s per B, plus per-sequence latency.

Usage: python src/jouleai/native/batch_bench.py [B_max]
"""

import ctypes
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.native.decoder3 import NativeDecoder  # noqa: E402

B_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10   # batch decode steps
WARMUP = 2

d = NativeDecoder("models/Qwen3-30B-A3B-Instruct-2507", max_tokens=512)
print(f"cfg L={d.cfg.L} d={d.cfg.d} V={d.cfg.V} E={d.cfg.E} topk={d.cfg.topk} "
      f"pool_threads={d._pool_threads}", flush=True)

toks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def run_batch(B: int, steps: int):
    d._seq_kv = {}
    d._seq_pos = {}
    h = np.empty(B * d.cfg.d, np.float32)
    lg = np.empty(B * d.cfg.V, np.float32)
    # embed the first tokens
    for b in range(B):
        d._dll.embed_lookup(d._w_ptr(), ctypes.pointer(d.cfg), toks[b],
                            ctypes.c_void_p(h.ctypes.data + b * d.cfg.d * 4))
    # decode step 0 (writes KV at pos 0), then steps 1..N-1 at pos=step
    for step in range(steps):
        pos_arr = (ctypes.c_int * B)(*([step] * B))
        d._dll.decode_layers_batch(
            ctypes.pointer(d.cfg), d._w_ptr(), d._kv_batch_ptr(B),
            pos_arr, ctypes.c_void_p(h.ctypes.data), B,
            ctypes.c_void_p(lg.ctypes.data))
        # feed next tokens: use argmax as the next token (keeps h consistent)
        for b in range(B):
            nxt = int(lg[b * d.cfg.V:(b + 1) * d.cfg.V].argmax())
            d._dll.embed_lookup(d._w_ptr(), ctypes.pointer(d.cfg), nxt,
                                ctypes.c_void_p(h.ctypes.data + b * d.cfg.d * 4))
    return h


print(f"{'B':>3} {'tok/s(agg)':>12} {'ms/step':>10} {'tok/s/seq':>10}", flush=True)
for B in range(1, B_MAX + 1):
    for _ in range(WARMUP):
        run_batch(B, 2)
    t0 = time.perf_counter()
    run_batch(B, STEPS)
    dt = time.perf_counter() - t0
    agg = B * STEPS / dt
    ms = dt / STEPS * 1000
    per_seq = agg / B
    print(f"{B:>3} {agg:>12.1f} {ms:>10.1f} {per_seq:>10.2f}", flush=True)
