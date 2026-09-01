"""Native full-decode harness: loads fixed weights (bf16) + resident Q4 experts
into memory and runs one C call per generated token (48 layers fused).

Physics: fixed-set reads ~2.9GB/token (attn+lm_head+embed+gate, bf16) +
experts ~1.15GB/token (Q4) — memory-bound. Removing Python per-layer overhead
is what converts the 2 tok/s wall into bandwidth-bound decode.

The KernelCfg is built from the ARCH REGISTRY (get_spec), so the kernel runs
any registered family (qwen2/qwen3/llama/mistral dense; olmoe/qwen3_moe/
mixtral MoE) with the right flags: qk_norm type, QKV bias, top-k renorm,
router naming, RoPE scaling (llama3). Unimplemented families (deepseek MLA,
gemma, phi, gpt_oss) raise loudly instead of routing wrong.
"""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.arch.registry import get_spec  # noqa: E402
from jouleai.native.backend import get_dll  # noqa: E402
from jouleai.storage.weight_store import SenseWeightStore  # noqa: E402

_DLL = get_dll("decode_kernel")


class KernelCfg(ctypes.Structure):
    _fields_ = [("L", ctypes.c_int), ("d", ctypes.c_int),
                ("n_heads", ctypes.c_int), ("n_kv", ctypes.c_int),
                ("hd", ctypes.c_int), ("E", ctypes.c_int),
                ("topk", ctypes.c_int), ("V", ctypes.c_int),
                ("eps", ctypes.c_float), ("maxT", ctypes.c_int),
                ("intermediate", ctypes.c_int),
                ("qk_norm_type", ctypes.c_int),
                ("bias_qkv", ctypes.c_int),
                ("norm_topk_prob", ctypes.c_int),
                ("expert_i8", ctypes.c_int),
                ("expert_bf16", ctypes.c_int)]


class KernelW(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.c_void_p), ("lm_head", ctypes.c_void_p),
        ("final_norm", ctypes.c_void_p),
        ("norm1", ctypes.c_void_p), ("norm2", ctypes.c_void_p),
        ("wq", ctypes.c_void_p), ("wk", ctypes.c_void_p),
        ("wv", ctypes.c_void_p), ("wo", ctypes.c_void_p),
        ("bq", ctypes.c_void_p), ("bk", ctypes.c_void_p), ("bv", ctypes.c_void_p),
        ("qn", ctypes.c_void_p), ("kn", ctypes.c_void_p),
        ("gate_w", ctypes.c_void_p),
        ("w1", ctypes.c_void_p), ("w2", ctypes.c_void_p), ("w3", ctypes.c_void_p),
        ("wq_i8", ctypes.c_void_p), ("wq_i8s", ctypes.c_void_p),
        ("wk_i8", ctypes.c_void_p), ("wk_i8s", ctypes.c_void_p),
        ("wv_i8", ctypes.c_void_p), ("wv_i8s", ctypes.c_void_p),
        ("wo_i8", ctypes.c_void_p), ("wo_i8s", ctypes.c_void_p),
        ("lm_i8", ctypes.c_void_p), ("lm_i8s", ctypes.c_void_p),
        ("expert_pk", ctypes.c_void_p * 3),
        ("expert_sc", ctypes.c_void_p * 3),
        ("cos", ctypes.c_void_p), ("sin", ctypes.c_void_p),
        ("use_i8", ctypes.c_int),
    ]


def _check_native_capable(spec) -> None:
    """Loud-fail for archs the native kernel cannot run yet (Entry 56: the
    registry DETECTS more families than any engine implements)."""
    if spec.mla:
        raise ValueError(
            f"native kernel: {spec.model_type} uses DeepSeek MLA attention "
            f"(latent KV) — not implemented in decode_kernel.c yet")
    if spec.model_type in ("gemma", "phi", "gpt_oss"):
        raise ValueError(
            f"native kernel: {spec.model_type} is detected by the registry but "
            f"not implemented (gelu/MLP naming/norm style unsupported)")


class NativeDecoder:
    def __init__(self, model_dir: str | Path, max_tokens: int = 4096,
                 precision: str = "q4"):
        self.dir = Path(model_dir)
        self.precision = precision          # q4 (default) | i8 (int8 experts) | bf16 (exact)
        self.cfg_data = json.loads((self.dir / "config.json").read_text())
        c = self.cfg_data
        self.spec = get_spec(c)
        _check_native_capable(self.spec)
        s = self.spec
        self.cfg = KernelCfg(
            L=s.n_layers, d=s.d,
            n_heads=s.n_heads, n_kv=s.n_kv,
            hd=s.head_dim,
            E=s.n_experts, topk=s.top_k,
            V=s.vocab, eps=s.eps,
            maxT=max_tokens,
            intermediate=s.intermediate,
            qk_norm_type={"none": 0, "per_head": 1, "whole": 2}[s.qk_norm],
            bias_qkv=1 if s.bias_qkv else 0,
            norm_topk_prob=1 if s.norm_topk_prob else 0)
        self.store = SenseWeightStore(self.dir)
        # QKV bias presence is model-inherent, not config-asserted (registry
        # defaults qwen2 to bias; a qwen2 checkpoint without bias tensors
        # would otherwise KeyError on load)
        if s.bias_qkv:
            if "model.layers.0.self_attn.q_proj.bias" not in self.store.names():
                self.cfg.bias_qkv = 0
                print("  [native] registry said bias_qkv but checkpoint has no "
                      "q_proj.bias — running unbias", flush=True)
        self.precision = getattr(self, "precision", "q4")
        if self.precision == "i8":
            self.cfg.expert_i8 = 1
        elif self.precision == "bf16":
            self.cfg.expert_bf16 = 1
        self._load_fixed()
        if self.cfg.E > 0:
            self._load_experts()
        self._rope_tables(max_tokens)
        self._kv: list[tuple[np.ndarray, np.ndarray]] = [
            (np.zeros((max_tokens, self.cfg.n_kv, self.cfg.hd), np.float32),
             np.zeros((max_tokens, self.cfg.n_kv, self.cfg.hd), np.float32))
            for _ in range(self.cfg.L)]
        self.pos = 0
        self._seq_kv: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._seq_pos: dict[str, int] = {}
        self._dll = _DLL
        # persistent thread pool (ggml-style spin barrier) for all batched GEMMs
        self._dll.spin_pool_init.argtypes = [ctypes.c_int]
        self._dll.spin_pool_init.restype = ctypes.c_int
        self._pool_threads = 8
        if not self._dll.spin_pool_init(self._pool_threads):
            raise RuntimeError("spin_pool_init failed")
        self._dll.decode_layers.argtypes = [
            ctypes.POINTER(KernelCfg), ctypes.POINTER(KernelW),
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p]
        self._dll.decode_layers.restype = None
        self._dll.set_lm_threads.argtypes = [ctypes.c_int]
        self._dll.set_lm_threads.restype = None
        self._dll.set_lm_threads(8)
        self._dll.decode_layers_batch.argtypes = [
            ctypes.POINTER(KernelCfg), ctypes.POINTER(KernelW),
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self._dll.decode_layers_batch.restype = None
        self._dll.decode_layers_batch_argmax.argtypes = [
            ctypes.POINTER(KernelCfg), ctypes.POINTER(KernelW),
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self._dll.decode_layers_batch_argmax.restype = None
        self._dll.embed_lookup.argtypes = [
            ctypes.POINTER(KernelW), ctypes.POINTER(KernelCfg),
            ctypes.c_int, ctypes.c_void_p]
        self._dll.embed_lookup.restype = None
        self._dll.debug_decode_layers.argtypes = [
            ctypes.POINTER(KernelCfg), ctypes.POINTER(KernelW),
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self._dll.debug_decode_layers.restype = None

    # ---------------- fixed weights (bf16 / int8, resident) --------
    def _cache_dir(self) -> Path:
        d = ROOT / "storage" / "converted" / self.dir.name / "fixed"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_or_convert(self, name: str, as_i8: bool):
        """Load a converted fixed weight from the .npy cache, or convert +
        cache it. Skips the 61GB safetensors re-read on subsequent startups."""
        import hashlib
        key = hashlib.md5(name.encode()).hexdigest()[:12]
        cdir = self._cache_dir()
        tag = "i8" if as_i8 else "bf16"
        w_path = cdir / f"{key}_{tag}.npy"
        s_path = cdir / f"{key}_{tag}_s.npy"
        if w_path.exists():
            w = np.load(w_path)
            if s_path.exists():
                return w, np.load(s_path)
            return w, None
        if as_i8:
            w, s = self._i8(name)
            np.save(w_path, w)
            np.save(s_path, s)
            return w, s
        w = self._bf16(name)
        np.save(w_path, w)
        return w, None

    def _bf16(self, name: str) -> np.ndarray:
        """Fixed weights as bf16 uint16 numpy (half the RAM/bandwidth of fp32;
        the C kernel dequantizes bf16->fp32 in registers)."""
        t = self.store.full(name)
        return t.view(torch.uint16).cpu().numpy().reshape(t.shape).copy()  # bf16 bits

    def _i8(self, name: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Fixed weight -> (unsigned int8 biased +128, per-row fp32 scale).

        Q8_0-style: per-row scale = max_abs/127; int8 = round(w/scale), stored
        as unsigned (bias +128) so the C kernel can use vpmaddubsw (u8 x s8)
        and subtract the bias. Returns (Wu [shape], scale [rows]).
        """
        t = self.store.full(name).float().numpy()
        flat = t.reshape(t.shape[0], -1)  # rows x cols
        amax = np.abs(flat).max(axis=1, keepdims=True)
        amax = np.maximum(amax, 1e-12)
        sc = (amax / 127.0).astype(np.float32)
        q = np.clip(np.round(flat / sc), -127, 127).astype(np.int16)
        u = (q + 128).astype(np.uint8).reshape(t.shape)
        return u, sc.ravel()

    def _load_fixed(self):
        d = self.cfg.d
        self.embed = self._bf16("model.embed_tokens.weight")
        # lm_head is the single biggest read (V×d); the bf16 dequant ALU cost
        # exceeds the bandwidth saving, so keep it fp32 (direct load + FMA).
        # Tied embeddings (tie_word_embeddings): lm_head = embed (bf16 bits).
        import json as _json
        _cfg0 = _json.loads((self.dir / "config.json").read_text())
        self.tied = bool(_cfg0.get("tie_word_embeddings", False))
        if self.tied:
            # lm_head = embed, but the C lm_head reads fp32 — convert embed to
            # fp32 for the head (the C embed_lookup still uses the bf16 copy).
            self.lm_head = self.store.full("model.embed_tokens.weight").float().numpy()
        else:
            self.lm_head = self.store.full("lm_head.weight").float().numpy()
        self.final_norm = self._bf16("model.norm.weight")
        n1 = np.empty((self.cfg.L, d), np.uint16)
        n2 = np.empty((self.cfg.L, d), np.uint16)
        hhd = self.cfg.n_heads * self.cfg.hd
        khd = self.cfg.n_kv * self.cfg.hd
        wq = np.empty((self.cfg.L, hhd, d), np.uint16)
        wk = np.empty((self.cfg.L, khd, d), np.uint16)
        wv = np.empty((self.cfg.L, khd, d), np.uint16)
        wo = np.empty((self.cfg.L, d, hhd), np.uint16)
        # QKV bias (qwen2 family) — fp32, [L, rows]
        bq = np.zeros((self.cfg.L, hhd), np.float32)
        bk = np.zeros((self.cfg.L, khd), np.float32)
        bv = np.zeros((self.cfg.L, khd), np.float32)
        # q_norm/k_norm: per_head (qwen3) weights are [hd]; whole (olmoe)
        # weights are the full vector [H*hd] / [KH*hd] — size from the
        # checkpoint so the C whole-vector path reads the right length.
        qn_rows = hhd if self.spec.qk_norm == "whole" else self.cfg.hd
        kn_rows = khd if self.spec.qk_norm == "whole" else self.cfg.hd
        qn = np.zeros((self.cfg.L, qn_rows), np.uint16)
        kn = np.zeros((self.cfg.L, kn_rows), np.uint16)
        self.dense = (self.cfg.E == 0)
        if self.dense:
            self.w1 = np.empty((self.cfg.L, self.cfg.intermediate, d), np.uint16)
            self.w2 = np.empty((self.cfg.L, self.cfg.intermediate, d), np.uint16)
            self.w3 = np.empty((self.cfg.L, d, self.cfg.intermediate), np.uint16)
            gw = np.empty((0, 0, 0), np.uint16)
        else:
            self.w1 = self.w2 = self.w3 = None
            gw = np.empty((self.cfg.L, self.cfg.E, d), np.uint16)
        qk_norm = self.spec.qk_norm
        for l in range(self.cfg.L):
            p = f"model.layers.{l}"
            n1[l] = self._load_or_convert(f"{p}.input_layernorm.weight", False)[0]
            n2[l] = self._load_or_convert(f"{p}.post_attention_layernorm.weight", False)[0]
            wq[l] = self._load_or_convert(f"{p}.self_attn.q_proj.weight", False)[0]
            wk[l] = self._load_or_convert(f"{p}.self_attn.k_proj.weight", False)[0]
            wv[l] = self._load_or_convert(f"{p}.self_attn.v_proj.weight", False)[0]
            wo[l] = self._load_or_convert(f"{p}.self_attn.o_proj.weight", False)[0]
            if qk_norm != "none":
                qn[l] = self._load_or_convert(f"{p}.self_attn.q_norm.weight", False)[0]
                kn[l] = self._load_or_convert(f"{p}.self_attn.k_norm.weight", False)[0]
            if self.cfg.bias_qkv:
                bq[l] = self.store.full(f"{p}.self_attn.q_proj.bias").float().numpy()
                bk[l] = self.store.full(f"{p}.self_attn.k_proj.bias").float().numpy()
                bv[l] = self.store.full(f"{p}.self_attn.v_proj.bias").float().numpy()
            if self.dense:
                # dense FFN: gate_proj/up_proj/down_proj
                self.w1[l] = self._load_or_convert(f"{p}.mlp.gate_proj.weight", False)[0]
                self.w2[l] = self._load_or_convert(f"{p}.mlp.up_proj.weight", False)[0]
                self.w3[l] = self._load_or_convert(f"{p}.mlp.down_proj.weight", False)[0]
            else:
                gw[l] = self._load_or_convert(f"{p}.{self._router_name()}.weight", False)[0]
        self.norm1, self.norm2 = n1, n2
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.bq, self.bk, self.bv = bq, bk, bv
        self.qn, self.kn, self.gate_w = qn, kn, gw
        # int8 Q8_0 attention variant (optional, built lazily)
        self._i8_wq = self._i8_wk = self._i8_wv = self._i8_wo = None
        self._i8_wq_s = self._i8_wk_s = self._i8_wv_s = self._i8_wo_s = None

    def _router_name(self) -> str:
        """MoE router tensor (no .weight suffix): qwen/olmoe use mlp.gate;
        mixtral/deepseek (block_sparse_moe) use mlp.block_sparse_moe.router."""
        if self.spec.expert_naming == "block_sparse_moe":
            return "mlp.block_sparse_moe.router"
        return "mlp.gate"

    def build_int8_attn(self):
        """Quantize attention QKV+o to int8 (per-row scale, unsigned bias).
        Enables the AVX-512 VNNI path (8x on QKV). Only on CPUs with VNNI.
        Uses the .npy cache — the 61GB safetensors conversion runs once."""
        if self._i8_wq is not None:
            return
        hhd = self.cfg.n_heads * self.cfg.hd
        khd = self.cfg.n_kv * self.cfg.hd
        d = self.cfg.d
        q_layers = []
        for l in range(self.cfg.L):
            p = f"model.layers.{l}"
            u, s = self._load_or_convert(f"{p}.self_attn.q_proj.weight", True)
            q_layers.append((u, s))
        # stack [L, H*hd, d] + scales [L, H*hd]
        self._i8_wq = np.stack([u for u, _ in q_layers])
        self._i8_wq_s = np.stack([s for _, s in q_layers])
        def _mk(name, shape_rows):
            us = [self._load_or_convert(f"model.layers.{l}.self_attn.{name}", True)
                  for l in range(self.cfg.L)]
            return np.stack([u for u, _ in us]), np.stack([s for _, s in us])
        self._i8_wk, self._i8_wk_s = _mk("k_proj.weight", khd)
        self._i8_wv, self._i8_wv_s = _mk("v_proj.weight", khd)
        self._i8_wo, self._i8_wo_s = _mk("o_proj.weight", d)
        # lm_head int8 (the biggest single read — V x d); tied => quantize embed
        if getattr(self, "tied", False):
            self._i8_lm, self._i8_lm_s = self._i8("model.embed_tokens.weight")
        else:
            self._i8_lm, self._i8_lm_s = self._load_or_convert("lm_head.weight", True)

    # ---------------- experts (mmap-lazy: Q4, int8, or bf16-exact) ---------
    def _load_experts(self):
        naming = self.spec.expert_naming
        if self.precision == "bf16":
            self._load_experts_bf16(naming)
            return
        from jouleai.storage.q4_store import convert_experts_i8, convert_experts_q4
        q4dir = ROOT / "storage" / "converted" / self.dir.name
        naming = self.spec.expert_naming
        if self.precision == "i8":
            # int8 Q8_0 experts: per-row fp32 scale, +128 bias (VNNI FFN,
            # ~1.5% rms error vs Q4's ~11% — the long-gen drift fix).
            bin_path = convert_experts_i8(self.dir, q4dir, self.cfg.L,
                                          self.cfg.E, naming=naming)
            idx_name = "experts_i8.json"
        else:
            bin_path = convert_experts_q4(self.dir, q4dir, self.cfg.L,
                                          self.cfg.E, naming=naming)
            idx_name = "experts_q4.json"
        idx = json.loads((q4dir / idx_name).read_text())
        # mmap is LAZY: pages fault in on first access (the "load on demand").
        # No pre-touch — RAM stays ∝ working set (active experts only). The OS
        # page cache holds hot experts; under pressure the OS reclaims them.
        self._mm = np.memmap(bin_path, dtype=np.uint8, mode="r")
        mm = self._mm
        n_e = self.cfg.L * self.cfg.E
        self.pk = [[ctypes.c_void_p(0)] * n_e for _ in range(3)]
        self.sc = [[ctypes.c_void_p(0)] * n_e for _ in range(3)]
        for l in range(self.cfg.L):
            for e in range(self.cfg.E):
                for p_i, part in enumerate(("gate", "up", "down")):
                    rec = idx[f"{l}.{e}.{part}"]
                    off = rec["offset"]
                    sb, pb = rec["scales_bytes"], rec["packed_bytes"]
                    key = (l * self.cfg.E + e)
                    self.pk[p_i][key] = ctypes.c_void_p(
                        mm.ctypes.data + off + sb)
                    self.sc[p_i][key] = ctypes.c_void_p(mm.ctypes.data + off)
        self.pk_arr = [(ctypes.c_void_p * n_e)(*a) for a in self.pk]
        self.sc_arr = [(ctypes.c_void_p * n_e)(*a) for a in self.sc]

    def _load_experts_bf16(self, naming: str):
        """bf16-exact expert tier: load the bf16 expert tensors directly (no
        quantization). The kernel reads them as uint16 bf16 rows; FFN math is
        exact (64-token token-identity — the drift fix). RAM = 2x the Q4
        store (bf16 vs int4); on this machine OLMoE's 6.5GB store fits."""
        n_e = self.cfg.L * self.cfg.E
        d, m = self.cfg.d, self.cfg.intermediate
        # per-part [L*E, rows, cols] bf16 — resident numpy uint16 views
        part_rows = {"gate": m, "up": m, "down": d}
        part_cols = {"gate": d, "up": d, "down": m}
        self._bf16_exp = {}
        for p_i, part in enumerate(("gate", "up", "down")):
            buf = np.empty((n_e, part_rows[part], part_cols[part]), np.uint16)
            for l in range(self.cfg.L):
                for e in range(self.cfg.E):
                    name = (f"model.layers.{l}.mlp.experts.{e}.{part}_proj.weight"
                            if naming == "qwen" else
                            f"model.layers.{l}.mlp.block_sparse_moe.experts."
                            f"{e}.{ {'gate':'w1','up':'w2','down':'w3'}[part] }.weight")
                    t = self.store.full(name)
                    buf[l * self.cfg.E + e] = t.view(torch.uint16).cpu().numpy()
            self._bf16_exp[part] = buf
        self.pk = [[ctypes.c_void_p(0)] * n_e for _ in range(3)]
        self.sc = [[ctypes.c_void_p(0)] * n_e for _ in range(3)]
        for p_i, part in enumerate(("gate", "up", "down")):
            for key in range(n_e):
                self.pk[p_i][key] = ctypes.c_void_p(
                    self._bf16_exp[part][key].ctypes.data)
                self.sc[p_i][key] = ctypes.c_void_p(0)
        self.pk_arr = [(ctypes.c_void_p * n_e)(*a) for a in self.pk]
        self.sc_arr = [(ctypes.c_void_p * n_e)(*a) for a in self.sc]

    # ---------------- rope tables -------------------------------------------
    def _rope_tables(self, maxT):
        s = self.spec
        hd = self.cfg.hd
        if s.rope_scaling and s.rope_scaling.get("rope_type") == "llama3":
            from jouleai.engine.generic_streamer import llama3_inv_freq
            inv = llama3_inv_freq(hd, s.theta, s.rope_scaling).numpy()
        else:
            theta = self.cfg_data.get("rope_theta", 1e4)
            inv = 1.0 / (theta ** (np.arange(0, hd, 2).astype(np.float64) / hd))
        pos = np.arange(maxT).astype(np.float64)[:, None]
        fr = pos * inv[None, :]
        emb = np.concatenate([fr, fr], axis=-1).astype(np.float32)
        self.cos = np.cos(emb).ravel()
        self.sin = np.sin(emb).ravel()

    # ---------------- forward -----------------------------------------------
    def _w_ptr(self) -> ctypes.POINTER(KernelW):
        w = KernelW()
        def P(a: np.ndarray) -> ctypes.c_void_p:
            return ctypes.c_void_p(a.ctypes.data)
        w.embed = P(self.embed)
        w.lm_head = P(self.lm_head)
        w.final_norm = P(self.final_norm)
        w.norm1 = P(self.norm1); w.norm2 = P(self.norm2)
        w.wq = P(self.wq); w.wk = P(self.wk); w.wv = P(self.wv); w.wo = P(self.wo)
        w.bq = P(self.bq); w.bk = P(self.bk); w.bv = P(self.bv)
        w.qn = P(self.qn); w.kn = P(self.kn); w.gate_w = P(self.gate_w)
        if getattr(self, "dense", False):
            w.w1 = P(self.w1); w.w2 = P(self.w2); w.w3 = P(self.w3)
            w.expert_pk[0] = w.expert_pk[1] = w.expert_pk[2] = ctypes.c_void_p(0)
            w.expert_sc[0] = w.expert_sc[1] = w.expert_sc[2] = ctypes.c_void_p(0)
        else:
            w.w1 = w.w2 = w.w3 = ctypes.c_void_p(0)
            w.expert_pk[0] = ctypes.c_void_p(ctypes.addressof(self.pk_arr[0]))
            w.expert_pk[1] = ctypes.c_void_p(ctypes.addressof(self.pk_arr[1]))
            w.expert_pk[2] = ctypes.c_void_p(ctypes.addressof(self.pk_arr[2]))
            w.expert_sc[0] = ctypes.c_void_p(ctypes.addressof(self.sc_arr[0]))
            w.expert_sc[1] = ctypes.c_void_p(ctypes.addressof(self.sc_arr[1]))
            w.expert_sc[2] = ctypes.c_void_p(ctypes.addressof(self.sc_arr[2]))
        # int8 Q8_0 attention path (if built)
        w.use_i8 = 1 if getattr(self, "_i8_wq", None) is not None else 0
        if w.use_i8:
            w.wq_i8 = P(self._i8_wq); w.wq_i8s = P(self._i8_wq_s)
            w.wk_i8 = P(self._i8_wk); w.wk_i8s = P(self._i8_wk_s)
            w.wv_i8 = P(self._i8_wv); w.wv_i8s = P(self._i8_wv_s)
            w.wo_i8 = P(self._i8_wo); w.wo_i8s = P(self._i8_wo_s)
            w.lm_i8 = P(self._i8_lm); w.lm_i8s = P(self._i8_lm_s)
        w.cos = P(self.cos); w.sin = P(self.sin)
        return ctypes.pointer(w)

    def _kv_ptr(self) -> ctypes.c_void_p:
        # array of KVCache { float* k; float* v; } per layer — keep alive on self
        self._kvc = (ctypes.c_void_p * (self.cfg.L * 2))()
        for l in range(self.cfg.L):
            self._kvc[l * 2] = ctypes.c_void_p(self._kv[l][0].ctypes.data)
            self._kvc[l * 2 + 1] = ctypes.c_void_p(self._kv[l][1].ctypes.data)
        return ctypes.cast(self._kvc, ctypes.c_void_p)

    class _KVCache(ctypes.Structure):
        _fields_ = [("k", ctypes.c_void_p), ("v", ctypes.c_void_p)]

    def _kv_ptrs(self, keys: list[str]) -> ctypes.c_void_p:
        """Per-seq KV as KVCache struct array over the given seq keys.

        Keys are the same the batch decode uses (seq0..seqB-1), so prefill
        and decode share KV seamlessly. Each sequence keeps ONE persistent KV
        pair per layer for the lifetime of the decoder (fresh zeroed buffers
        are created only when a sequence id is first seen or after reset).
        """
        L = self.cfg.L
        B = len(keys)
        self._kvc_b = (self._KVCache * (B * L))()
        for b, key in enumerate(keys):
            if key not in self._seq_kv:
                self._seq_kv[key] = [
                    (np.zeros((self.cfg.maxT, self.cfg.n_kv, self.cfg.hd), np.float32),
                     np.zeros((self.cfg.maxT, self.cfg.n_kv, self.cfg.hd), np.float32))
                    for _ in range(L)]
                self._seq_pos[key] = 0
            for l in range(L):
                kk, vv = self._seq_kv[key][l]
                self._kvc_b[b * L + l].k = ctypes.c_void_p(kk.ctypes.data)
                self._kvc_b[b * L + l].v = ctypes.c_void_p(vv.ctypes.data)
        return ctypes.cast(self._kvc_b, ctypes.c_void_p)

    def _kv_batch_ptr(self, B: int) -> ctypes.c_void_p:
        return self._kv_ptrs([f"seq{b}" for b in range(B)])

    def prefill(self, tokens: list[int]) -> np.ndarray:
        """Prefill a prompt [T tokens] via the C prefill kernel.

        Returns the first-token logits (fp32 [V]). The lm_head is applied
        only at the last token. KV is written into the batch KV (seq0) so
        decode_token/decode_batch continue seamlessly."""
        self._dll.prefill_layers.argtypes = [
            ctypes.POINTER(KernelCfg), ctypes.POINTER(KernelW),
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_void_p]
        self._dll.prefill_layers.restype = None
        T = len(tokens)
        tok_arr = (ctypes.c_int * T)(*tokens)
        logits = np.empty(self.cfg.V, np.float32)
        self._seq_kv = {}
        self._seq_pos = {}
        self._dll.prefill_layers(
            ctypes.pointer(self.cfg), self._w_ptr(), self._kv_batch_ptr(1),
            tok_arr, T, ctypes.c_void_p(logits.ctypes.data))
        self._seq_pos["seq0"] = T
        return logits

    def prefill_batch(self, prompts: list[list[int]]) -> list[np.ndarray]:
        """Batched prefill: B prompts through all layers TOGETHER.

        One decode_layers_batch per position t over the prompts still alive at
        t — weights are read once per position across the batch (the prefill
        amortization, the first-token-latency fix). lm_head is computed only
        when a position is the last token of every alive prompt (equal-length
        prompts: one lm_head pass for the whole batch); unequal-length
        stragglers get a final single decode at their last position.

        KV is written under keys seq0..seqB-1 (the same keys decode_batch
        uses), so prefill_batch -> decode_batch is seamless.
        """
        B = len(prompts)
        self._seq_kv = {}
        self._seq_pos = {}
        d, V = self.cfg.d, self.cfg.V
        maxT = max(len(p) for p in prompts)
        outs: list[np.ndarray | None] = [None] * B
        for t in range(maxT):
            active = [b for b in range(B) if t < len(prompts[b])]
            Bt = len(active)
            if Bt == 0:
                break
            h = np.empty(Bt * d, np.float32)
            for i, b in enumerate(active):
                self._dll.embed_lookup(self._w_ptr(), ctypes.pointer(self.cfg),
                                       prompts[b][t],
                                       ctypes.c_void_p(h.ctypes.data + i * d * 4))
            pos_arr = (ctypes.c_int * Bt)(*([t] * Bt))
            all_last = all(t == len(prompts[b]) - 1 for b in active)
            lg = np.empty(Bt * V, np.float32) if all_last else None
            self._dll.decode_layers_batch(
                ctypes.pointer(self.cfg), self._w_ptr(),
                self._kv_ptrs([f"seq{b}" for b in active]),
                pos_arr, ctypes.c_void_p(h.ctypes.data), Bt,
                ctypes.c_void_p(lg.ctypes.data) if lg is not None else None)
            if all_last:
                for i, b in enumerate(active):
                    outs[b] = lg[i * V:(i + 1) * V]
        # unequal-length stragglers: one final decode at their last position
        # (recomputes that position, which the main loop already stored in KV)
        for b in range(B):
            if outs[b] is None:
                T = len(prompts[b])
                h = np.empty(d, np.float32)
                self._dll.embed_lookup(self._w_ptr(), ctypes.pointer(self.cfg),
                                       prompts[b][-1],
                                       ctypes.c_void_p(h.ctypes.data))
                pos_arr = (ctypes.c_int * 1)(T - 1)
                lg = np.empty(V, np.float32)
                self._dll.decode_layers_batch(
                    ctypes.pointer(self.cfg), self._w_ptr(),
                    self._kv_ptrs([f"seq{b}"]),
                    pos_arr, ctypes.c_void_p(h.ctypes.data), 1,
                    ctypes.c_void_p(lg.ctypes.data))
                outs[b] = lg
        for b in range(B):
            self._seq_pos[f"seq{b}"] = len(prompts[b])
        return outs

    def decode_token(self, token: int) -> int:
        """Decode one token (KV at self.pos), returns next token argmax.

        Uses the BATCH kernel with B=1 (pool-parallel lm_head — faster than
        the thread-per-call single path)."""
        return self.decode_batch([token], [self._seq_pos.get("seq0", 0)])[0]

    def decode_batch(self, tokens: list[int], positions: list[int]) -> list[int]:
        """Batch decode: B sequences (one token each), shared weight pass.

        positions = the KV slot each sequence writes at (== its current
        length). Each sequence's KV is persistent across calls. Returns the
        argmax next token per sequence.
        """
        B = len(tokens)
        if self._seq_kv is None:
            self._seq_kv = {}
            self._seq_pos = {}
        d = self.cfg.d
        h = np.empty(B * d, np.float32)
        for b, t in enumerate(tokens):
            self._dll.embed_lookup(self._w_ptr(), ctypes.pointer(self.cfg), t,
                                   ctypes.c_void_p(h.ctypes.data + b * d * 4))
        logits = np.empty(B * self.cfg.V, np.float32)
        pos_arr = (ctypes.c_int * B)(*positions)
        self._dll.decode_layers_batch(
            ctypes.pointer(self.cfg), self._w_ptr(), self._kv_batch_ptr(B),
            pos_arr, ctypes.c_void_p(h.ctypes.data), B,
            ctypes.c_void_p(logits.ctypes.data))
        for b in range(B):
            self._seq_pos[f"seq{b}"] = positions[b] + 1
        return [int(logits[b * self.cfg.V:(b + 1) * self.cfg.V].argmax())
                for b in range(B)]

    def decode_batch_argmax(self, tokens: list[int],
                            positions: list[int]) -> list[int]:
        """Batch decode, argmax-only: returns the next token per sequence
        WITHOUT materializing the B*V logits (the kernel computes the argmax
        in C). The hot serve decode path — skips the ~4.8MB numpy logits
        alloc + argmax per batch step."""
        B = len(tokens)
        if self._seq_kv is None:
            self._seq_kv = {}
            self._seq_pos = {}
        d = self.cfg.d
        h = np.empty(B * d, np.float32)
        for b, t in enumerate(tokens):
            self._dll.embed_lookup(self._w_ptr(), ctypes.pointer(self.cfg), t,
                                   ctypes.c_void_p(h.ctypes.data + b * d * 4))
        out = (ctypes.c_int * B)()
        pos_arr = (ctypes.c_int * B)(*positions)
        self._dll.decode_layers_batch_argmax(
            ctypes.pointer(self.cfg), self._w_ptr(), self._kv_batch_ptr(B),
            pos_arr, ctypes.c_void_p(h.ctypes.data), B, out)
        for b in range(B):
            self._seq_pos[f"seq{b}"] = positions[b] + 1
        return list(out)

    def decode_spec_verify(self, tokens: list[int], positions: list[int]) \
            -> list[np.ndarray]:
        """Spec-decode verify: gamma drafted tokens in ONE batch call over a
        SHARED KV (seq0) — the correct spec-decode semantics.

        Each draft token b decodes at positions[b] over KV = prefix +
        draft[0..b-1]. All B slots point at the SAME seq0 buffers; the kernel
        processes seqs in order per layer, so the shared buffer accumulates
        the draft K/V rows as the pass goes — exactly the sequential
        autoregressive forward, but the weight rows are read ONCE per layer
        for all gamma positions (the batch amortization that makes spec
        decode > plain decode).

        Returns logits[b] = the target's prediction AT position positions[b]
        (compare argmax(logits[b]) vs draft[b+1] for acceptance).
        """
        B = len(tokens)
        d, V = self.cfg.d, self.cfg.V
        key0 = "seq0"
        if key0 not in self._seq_kv:          # need a KV to verify against
            self._seq_kv[key0] = [
                (np.zeros((self.cfg.maxT, self.cfg.n_kv, self.cfg.hd), np.float32),
                 np.zeros((self.cfg.maxT, self.cfg.n_kv, self.cfg.hd), np.float32))
                for _ in range(self.cfg.L)]
            self._seq_pos[key0] = 0
        h = np.empty(B * d, np.float32)
        for i, t in enumerate(tokens):
            self._dll.embed_lookup(self._w_ptr(), ctypes.pointer(self.cfg), t,
                                   ctypes.c_void_p(h.ctypes.data + i * d * 4))
        L = self.cfg.L
        self._kvc_b = (self._KVCache * (B * L))()
        for b in range(B):                    # all slots -> the SAME seq0 KV
            for l in range(L):
                kk, vv = self._seq_kv[key0][l]
                self._kvc_b[b * L + l].k = ctypes.c_void_p(kk.ctypes.data)
                self._kvc_b[b * L + l].v = ctypes.c_void_p(vv.ctypes.data)
        pos_arr = (ctypes.c_int * B)(*positions)
        lg = np.empty(B * V, np.float32)
        self._dll.decode_layers_batch(
            ctypes.pointer(self.cfg), self._w_ptr(),
            ctypes.cast(self._kvc_b, ctypes.c_void_p),
            pos_arr, ctypes.c_void_p(h.ctypes.data), B,
            ctypes.c_void_p(lg.ctypes.data))
        self._seq_pos[key0] = positions[-1] + 1
        return [lg[i * V:(i + 1) * V] for i in range(B)]

    def reset(self):
        for l in range(self.cfg.L):
            self._kv[l][0].fill(0)
            self._kv[l][1].fill(0)
        self.pos = 0
        self._seq_kv = {}
        self._seq_pos = {}

    def debug_decode_layers(self, pos: int, token: int) -> np.ndarray:
        """Debug: capture per-layer decode intermediates at `pos` (KV must
        already hold positions < pos). Returns [L, 7, max(d,E)] fp32 rows:
        attn_in, attn_out, h_after_attn, ffn_in, router_logits, ffn_out,
        h_after_ffn. Diagnostic only (drift bisect)."""
        L, d, E = self.cfg.L, self.cfg.d, self.cfg.E
        rowlen = max(d, E)
        h = np.empty(d, np.float32)
        self._dll.embed_lookup(self._w_ptr(), ctypes.pointer(self.cfg), token,
                               ctypes.c_void_p(h.ctypes.data))
        out = np.empty(L * 7 * rowlen, np.float32)
        self._dll.debug_decode_layers(
            ctypes.pointer(self.cfg), self._w_ptr(), self._kv_batch_ptr(1),
            pos, ctypes.c_void_p(h.ctypes.data), ctypes.c_void_p(out.ctypes.data))
        return out.reshape(L, 7, rowlen)

    # ---------------- resident working set (RAM ∝ what's used) -------------
    def resident_bytes(self) -> int:
        """Bytes of the expert mmap actually resident in RAM (page cache)."""
        try:
            import mmap as _mm
            # count resident pages via the memmap's private mapping is not
            # directly exposed; approximate with the touched range is complex.
            # Use psutil RSS delta vs the mmap length as the honest proxy.
            import psutil
            return int(psutil.Process().memory_info().rss)
        except Exception:
            return 0

    def release(self, keep_mb: int = 0):
        """Release the expert page cache after a generation (the 'release after
        use'): advise the OS to drop pages beyond the hot set. Windows has no
        posix_fadvise; we rely on the OS LRU page cache under memory pressure.
        Explicit drop is best-effort via mmap.madvise where available."""
        try:
            import mmap as _mm
            if hasattr(_mm, "MADV_DONTNEED") and hasattr(self._mm, "madvise"):
                # keep the first `keep_mb` MB hot (recently used experts)
                keep = keep_mb * 1048576
                if keep < self._mm.size:
                    self._mm.madvise(keep, self._mm.size - keep, _mm.MADV_DONTNEED)
        except Exception:
            pass
