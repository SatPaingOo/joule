"""Q4 quantized expert store — 4x less disk and IO for MoE expert weights.

Format v1 (per tensor): groups of 64 values; per group one fp16 scale
(absmax/7); values quantised to int4 in [-8, 7], stored biased (+8), two per
byte (low nibble first). Record layout: [scales fp16][packed u8].

convert_experts_q4(): one-time converter pass over the safetensors experts.
Q4ExpertPool(): drop-in expert pool whose gathers read + unpack Q4 records;
IO accounting counts packed bytes (the bytes actually pulled from disk).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from jouleai.storage.expert_store import ExpertLRUPool  # noqa: F401  (re-export)
from jouleai.storage.weight_store import SenseWeightStore

GROUP = 64
PARTS = ("gate", "up", "down")


def expert_tensor_name(l: int, e: int, part: str, naming: str = "qwen") -> str:
    """Map (layer, expert, part) to the model's tensor name.

    qwen (Qwen2/3-MoE, OLMoE):  mlp.experts.{e}.{gate,up,down}_proj.weight
    block_sparse_moe (Mixtral, DeepSeek): mlp.block_sparse_moe.experts.{e}.{w1,w2,w3}.weight
    """
    if naming == "block_sparse_moe":
        part_map = {"gate": "w1", "up": "w2", "down": "w3"}
        return (f"model.layers.{l}.mlp.block_sparse_moe.experts.{e}."
                f"{part_map[part]}.weight")
    return f"model.layers.{l}.mlp.experts.{e}.{part}_proj.weight"


def _quantize_tensor(x: np.ndarray, group: int = GROUP) -> tuple[bytes, bytes]:
    """bf16/float tensor -> (scales_bytes fp16, packed_bytes uint4-biased)."""
    flat = x.astype(np.float32).ravel()
    n = flat.size
    pad = (-n) % group
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    g = flat.reshape(-1, group)
    absmax = np.maximum(np.abs(g).max(axis=1), 1e-12)
    scales = (absmax / 7.0).astype(np.float16)
    q = np.clip(np.round(g / scales.astype(np.float32)[:, None]), -8, 7).astype(np.int8)
    qb = (q + 8).astype(np.uint8).ravel()[:n]  # drop padding tail after clip
    if pad:
        qb = np.concatenate([qb, np.zeros(pad, np.uint8)])
    packed = (qb[0::2] & 0x0F) | (qb[1::2] << 4)
    return scales.tobytes(), packed.tobytes()


def _dequantize(scales_b: bytes, packed_b: bytes, numel: int,
                group: int = GROUP) -> np.ndarray:
    raw = np.frombuffer(packed_b, dtype=np.uint8)
    lo = (raw & 0x0F).astype(np.int16) - 8
    hi = (raw >> 4).astype(np.int16) - 8
    q = np.empty(raw.size * 2, dtype=np.int16)
    q[0::2] = lo
    q[1::2] = hi
    scales = np.frombuffer(scales_b, dtype=np.float16).astype(np.float32)
    return (q.reshape(-1, group) * scales[:, None]).ravel()[:numel]


def convert_experts_i8(model_dir: str | Path, out_dir: str | Path | None = None,
                       n_layers: int | None = None, n_experts: int | None = None,
                       progress_every: int = 512, naming: str = "qwen") -> Path:
    """One-time conversion of all expert tensors to an int8 store (VNNI-ready).

    Q8_0-style: per-row scale (max_abs/127), int8 values stored unsigned
    (+128 bias) so the C kernel can use vpmaddubsw (u8 x s8). 2x the Q4 store
    size but enables AVX-512 VNNI in the FFN (the per-layer bottleneck).
    Record layout: [scales fp32 per row][packed int8 rows], scales first.
    """
    model_dir = Path(model_dir)
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[3] / "storage" / "converted" \
            / model_dir.name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path, idx_path = out_dir / "experts_i8.bin", out_dir / "experts_i8.json"
    if bin_path.exists() and idx_path.exists():
        return bin_path

    store = SenseWeightStore(model_dir)
    index: dict[str, dict] = {}
    offset = 0
    written = 0
    with open(bin_path, "wb") as f:
        for l in range(n_layers):
            for e in range(n_experts):
                for part in PARTS:
                    name = expert_tensor_name(l, e, part, naming)
                    x = store.full(name).float().numpy()
                    rows = x.reshape(x.shape[0], -1)
                    amax = np.maximum(np.abs(rows).max(axis=1, keepdims=True), 1e-12)
                    sc = (amax / 127.0).astype(np.float32)
                    q = np.clip(np.round(rows / sc), -127, 127).astype(np.int16)
                    u = (q + 128).astype(np.uint8)
                    sc_b = sc.tobytes()
                    pk_b = u.tobytes()
                    f.write(sc_b)
                    f.write(pk_b)
                    index[f"{l}.{e}.{part}"] = {
                        "offset": offset, "numel": int(x.size),
                        "scales_bytes": len(sc_b), "packed_bytes": len(pk_b),
                        "shape": list(x.shape),
                    }
                    offset += len(sc_b) + len(pk_b)
                    written += 1
                    if written % progress_every == 0:
                        print(f"  i8 convert: {written} tensors, "
                              f"{offset / 1e9:.1f} GB", flush=True)
    idx_path.write_text(json.dumps(index))
    print(f"i8 expert store ready: {bin_path} ({bin_path.stat().st_size / 1e9:.1f} GB)",
          flush=True)
    return bin_path


def convert_experts_q4(model_dir: str | Path, out_dir: str | Path | None = None,
                       n_layers: int | None = None, n_experts: int | None = None,
                       progress_every: int = 512, naming: str = "qwen") -> Path:
    """One-time conversion of all expert tensors to the packed Q4 file."""
    model_dir = Path(model_dir)
    if out_dir is None:
        out_dir = Path(__file__).resolve().parents[3] / "storage" / "converted" \
            / model_dir.name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path, idx_path = out_dir / "experts_q4.bin", out_dir / "experts_q4.json"
    if bin_path.exists() and idx_path.exists():
        return bin_path

    store = SenseWeightStore(model_dir)
    index: dict[str, dict] = {}
    offset = 0
    written = 0
    with open(bin_path, "wb") as f:
        for l in range(n_layers):
            for e in range(n_experts):
                for part in PARTS:
                    name = expert_tensor_name(l, e, part, naming)
                    x = store.full(name).float().numpy()
                    scales_b, packed_b = _quantize_tensor(x)
                    f.write(scales_b)
                    f.write(packed_b)
                    index[f"{l}.{e}.{part}"] = {
                        "offset": offset, "numel": int(x.size),
                        "scales_bytes": len(scales_b),
                        "packed_bytes": len(packed_b),
                        "shape": list(store.shape_of(name)),
                    }
                    offset += len(scales_b) + len(packed_b)
                    written += 1
                    if written % progress_every == 0:
                        print(f"  q4 convert: {written} tensors, "
                              f"{offset / 1e9:.1f} GB", flush=True)
    idx_path.write_text(json.dumps(index))
    print(f"q4 store ready: {bin_path} ({bin_path.stat().st_size / 1e9:.1f} GB)",
          flush=True)
    return bin_path


class Q4ExpertPool:
    """Expert pool reading packed-Q4 records.

    dequant=True (default): gather() returns bf16 tensors (numpy unpack).
    raw=True: gather() returns (scales_view, packed_view) per part — no
    dequantisation; the native kernel consumes these directly (÷4 RAM, ÷4 IO).
    """

    def __init__(self, model_dir: str | Path, n_layers: int, n_experts: int,
                 budget_bytes: int, converted_dir: str | Path | None = None,
                 raw: bool = False):
        model_dir = Path(model_dir)
        cdir = Path(converted_dir) if converted_dir else \
            Path(__file__).resolve().parents[3] / "storage" / "converted" / model_dir.name
        self.raw = raw
        self.bin_path = convert_experts_q4(model_dir, cdir, n_layers, n_experts)
        self.index = json.loads((cdir / "experts_q4.json").read_text())
        self._mm = np.memmap(self.bin_path, dtype=np.uint8, mode="r")
        self.store = SenseWeightStore(model_dir)  # for fixed weights elsewhere
        self._inner = ExpertLRUPool(self, budget_bytes)

    def expert_size_bytes(self, l: int, e: int) -> int:
        return sum(self.index[f"{l}.{e}.{p}"]["scales_bytes"]
                   + self.index[f"{l}.{e}.{p}"]["packed_bytes"] for p in PARTS)

    def gather(self, l: int, e: int):
        outs = []
        for part in PARTS:
            rec = self.index[f"{l}.{e}.{part}"]
            o = rec["offset"]
            sb, pb = rec["scales_bytes"], rec["packed_bytes"]
            scales = np.frombuffer(self._mm[o:o + sb].tobytes(), dtype=np.uint16)
            packed = np.frombuffer(self._mm[o + sb:o + sb + pb].tobytes(),
                                   dtype=np.uint8)
            if self.raw:
                outs.append((scales, packed, rec["numel"]))
            else:
                arr = _dequantize(bytes(scales.tobytes()), bytes(packed.tobytes()),
                                  rec["numel"])
                t = torch.from_numpy(arr).to(torch.bfloat16)
                outs.append(t.reshape(rec["shape"]))
        return tuple(outs)

    def __getattr__(self, name):
        return getattr(self._inner, name)
