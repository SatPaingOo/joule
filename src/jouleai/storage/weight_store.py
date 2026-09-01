"""SenseWeightStore — the model-as-database weight store.

Parses a safetensors file, memory-maps the data region, and serves tensors at
ROW granularity with zero-copy gathers: touched bytes ~= requested bytes.
BF16 payloads are stored as uint16 and reinterpreted via torch.view.

FFN neuron i (Qwen2-style) maps to:
  gate_proj row i, up_proj row i (shape [d_model]) and
  down_proj column i (i.e. row-major slice [:, i] of [d_model, d_ff]).
For column gathers we use strided views over the memmap so only the needed
columns' pages are touched.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_DTYPES = {
    "BF16": (np.uint16, torch.bfloat16, 2),
    "F16": (np.float16, torch.float16, 2),
    "F32": (np.float32, torch.float32, 4),
}


@dataclass
class TensorMeta:
    name: str
    shape: tuple[int, ...]
    np_dtype: np.dtype
    pt_dtype: torch.dtype
    item_size: int
    abs_start: int  # absolute byte offset in file
    abs_end: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


class SenseWeightStore:
    """mmap-backed, row-addressable view of a model directory (sharded or single).

    Accepts either a single .safetensors file or a model directory containing
    model.safetensors.index.json; tensor lookups are routed to the right shard.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        self._shards: dict[str, SenseWeightStore] = {}
        if path.is_file():
            self._init_single(path)
        else:
            single = path / "model.safetensors"
            if single.exists():
                self._init_single(single)
            else:
                self._init_sharded(path)

    def _init_single(self, file: Path) -> None:
        self.path = file
        self._is_sharded = False
        with open(file, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        self._data_start = 8 + hlen
        self.meta: dict[str, TensorMeta] = {}
        for name, t in header.items():
            if name == "__metadata__":
                continue
            np_dt, pt_dt, isz = _DTYPES[t["dtype"]]
            s, e = t["data_offsets"]
            self.meta[name] = TensorMeta(
                name=name, shape=tuple(t["shape"]), np_dtype=np.dtype(np_dt),
                pt_dtype=pt_dt, item_size=isz,
                abs_start=self._data_start + s, abs_end=self._data_start + e,
            )
        self._mm: np.memmap | None = None

    def _init_sharded(self, d: Path) -> None:
        idx = json.loads((d / "model.safetensors.index.json").read_text())
        self.path = d
        weight_map: dict[str, str] = idx["weight_map"]
        self._is_sharded = True
        self._weight_map = weight_map
        self.meta: dict[str, TensorMeta] = {}
        self._shard_paths: dict[str, Path] = {}
        for name, shard in sorted(weight_map.items()):
            if shard not in self._shard_paths:
                self._shard_paths[shard] = d / shard
        self._mm = None

    def _resolve(self, name: str) -> "SenseWeightStore":
        """Return the single-file store that owns `name` (lazy shard open)."""
        if self._is_sharded:
            shard = self._weight_map[name]
            if shard not in self._shards:
                self._shards[shard] = SenseWeightStore(self._shard_paths[shard])
            return self._shards[shard]
        return self

    @property
    def mm(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self._mm

    def names(self, pattern: str | None = None) -> list[str]:
        src = self._weight_map if self._is_sharded else self.meta
        if pattern is None:
            return sorted(src)
        return sorted(n for n in src if pattern in n)

    # -- full tensors (fixed weights: embed, attention, norms) --------------
    def full(self, name: str) -> torch.Tensor:
        return self._resolve(name)._local_full(name)

    def _local_full(self, name: str) -> torch.Tensor:
        m = self.meta[name]
        raw = self.mm[m.abs_start:m.abs_end]
        arr = np.frombuffer(raw, dtype=m.np_dtype).reshape(m.shape)
        t = torch.from_numpy(arr.copy())  # copy: fixed weights stay in RAM
        return t if m.pt_dtype != torch.bfloat16 else t.view(torch.bfloat16)

    def _view(self, m: TensorMeta, byte_start: int, shape: tuple[int, ...]) -> torch.Tensor:
        """Zero-copy bf16/f16 view; F32 needs an explicit cast by the caller."""
        n = 1
        for d in shape:
            n *= d
        raw = self.mm[byte_start:byte_start + n * m.item_size]
        u16 = np.frombuffer(raw, dtype=m.np_dtype)
        t = torch.from_numpy(u16)  # zero-copy (uint16/uint-based)
        t = t.view(m.pt_dtype) if m.pt_dtype == torch.bfloat16 else t.float()
        return t.reshape(shape)

    # -- row gathers (gate/up rows; down cols) -------------------------------
    def rows(self, name: str, idx: torch.Tensor | list[int]) -> torch.Tensor:
        """Gather rows of a 2-D tensor [d0, d1] by index -> [len(idx), d1].

        Only the requested rows' bytes are touched (zero-copy numpy strided read
        into one small copy for torch).
        """
        return self._resolve(name).local_rows(name, idx)

    def local_rows(self, name: str, idx: torch.Tensor | list[int]) -> torch.Tensor:
        m = self.meta[name]
        d0, d1 = m.shape
        row_bytes = d1 * m.item_size
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        starts = m.abs_start + (idx_t * row_bytes).numpy()
        out = np.empty((len(idx_t), d1), dtype=m.np_dtype)
        raw = self.mm
        for r, s in enumerate(starts):
            out[r] = np.frombuffer(raw[s:s + row_bytes], dtype=m.np_dtype)
        t = torch.from_numpy(out)
        return t.view(m.pt_dtype) if m.pt_dtype == torch.bfloat16 else t.float()

    def cols(self, name: str, idx: torch.Tensor | list[int]) -> torch.Tensor:
        """Gather columns of a 2-D tensor [d0, d1] -> [d0, len(idx)].

        NOTE: in the original row-major layout a column gather touches every
        row's pages, so for true neuron-granular access the Sense Layer
        Converter writes a transposed copy (see convert_down_t); gather
        columns of `name` via rows() on the transposed tensor instead.
        """
        return self._resolve(name).local_cols(name, idx)

    def local_cols(self, name: str, idx: torch.Tensor | list[int]) -> torch.Tensor:
        m = self.meta[name]
        d0, d1 = m.shape
        idx_np = torch.as_tensor(idx, dtype=torch.long).numpy()
        out = np.empty((d0, len(idx_np)), dtype=m.np_dtype)
        raw = self.mm
        row_stride = d1 * m.item_size
        base = m.abs_start
        for r in range(d0):
            row = np.frombuffer(raw[base + r * row_stride: base + (r + 1) * row_stride],
                                dtype=m.np_dtype)
            out[r] = row[idx_np]
        t = torch.from_numpy(out)
        return t.view(m.pt_dtype) if m.pt_dtype == torch.bfloat16 else t.float()

    def bytes_of(self, name: str) -> int:
        m = self._resolve(name).meta[name]
        return m.abs_end - m.abs_start

    def shape_of(self, name: str) -> tuple[int, ...]:
        return self._resolve(name).meta[name].shape

    def close(self) -> None:
        for s in self._shards.values():
            s.close()
        self._shards.clear()
        if getattr(self, "_mm", None) is not None:
            del self._mm
            self._mm = None
