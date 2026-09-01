"""GenericStreamer — one flag-driven forward for all registered families.

Covers: qwen2 / qwen3 / llama / mistral (dense) and olmoe / qwen3_moe /
mixtral (MoE streaming). Q4 expert pools and the native kernel attach
transparently for MoE models (same interface as the phase experiments).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from jouleai.arch.registry import ArchSpec, get_spec
from jouleai.storage.weight_store import SenseWeightStore


def llama3_inv_freq(dim: int, base: float, scaling: dict) -> torch.Tensor:
    """LLama3 scaled inv_freq (rope_type='llama3'; HF divides low freqs)."""
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    orig_max = float(scaling.get("original_max_position_tokens", 8192))
    factor = float(scaling.get("factor", 1.0))
    low_f = float(scaling.get("low_freq_factor", 1.0))
    high_f = float(scaling.get("high_freq_factor", 4.0))
    low_w, high_w = orig_max / low_f, orig_max / high_f
    out = inv.clone()
    for i, f in enumerate(inv.tolist()):
        wavelen = 2 * 3.141592653589793 / f
        if wavelen < high_w:
            out[i] = f                    # high-frequency: unchanged
        elif wavelen > low_w:
            out[i] = f / factor           # low-frequency: stretch (HF divides)
        else:
            smooth = (orig_max / wavelen - low_f) / (high_f - low_f)
            out[i] = (1 - smooth) * (f / factor) + smooth * f
    return out


def _check_implemented(spec) -> None:
    """Loud-fail for families the registry DETECTS but no engine implements
    (Entry 56): deepseek MLA, gemma/phi/gpt_oss naming+norm. The Python and
    native paths both gate here so a convert/serve on them never silently
    routes wrong."""
    if spec.mla:
        raise ValueError(
            f"{spec.model_type} uses DeepSeek MLA attention (latent KV) — "
            f"detected by the registry but not implemented in any engine")
    if spec.model_type in ("gemma", "phi", "gpt_oss"):
        raise ValueError(
            f"{spec.model_type} is detected by the registry but not "
            f"implemented (norm/MLP style unsupported)")


class GenericStreamer:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.spec: ArchSpec = get_spec(self.cfg)
        _check_implemented(self.spec)
        self.store = SenseWeightStore(self.dir)
        s = self.spec
        self.embed = self.store.full("model.embed_tokens.weight")
        self.lm_head = self.embed if s.tied else self.store.full("lm_head.weight")
        self.final_norm = self.store.full("model.norm.weight")
        self.wq, self.wk, self.wv, self.wo = [], [], [], []
        self.bq, self.bk, self.bv = [], [], []
        self.qn, self.kn = [], []
        self.norm1, self.norm2, self.gate = [], [], []
        for l in range(s.n_layers):
            p = f"model.layers.{l}"
            self.wq.append(self.store.full(f"{p}.self_attn.q_proj.weight"))
            self.wk.append(self.store.full(f"{p}.self_attn.k_proj.weight"))
            self.wv.append(self.store.full(f"{p}.self_attn.v_proj.weight"))
            self.wo.append(self.store.full(f"{p}.self_attn.o_proj.weight"))
            if s.bias_qkv:
                self.bq.append(self.store.full(f"{p}.self_attn.q_proj.bias"))
                self.bk.append(self.store.full(f"{p}.self_attn.k_proj.bias"))
                self.bv.append(self.store.full(f"{p}.self_attn.v_proj.bias"))
            if s.qk_norm != "none":
                self.qn.append(self.store.full(f"{p}.self_attn.q_norm.weight"))
                self.kn.append(self.store.full(f"{p}.self_attn.k_norm.weight"))
            self.norm1.append(self.store.full(f"{p}.input_layernorm.weight"))
            self.norm2.append(self.store.full(f"{p}.post_attention_layernorm.weight"))
            if s.moe:
                self.gate.append(self.store.full(f"{p}.mlp.gate.weight"))
        if s.rope_scaling and s.rope_scaling.get("rope_type") == "llama3":
            self.inv_freq = llama3_inv_freq(s.head_dim, s.theta, s.rope_scaling)
        else:
            self.inv_freq = 1.0 / (s.theta ** (torch.arange(0, s.head_dim, 2).float()
                                               / s.head_dim))
        self.dtype = torch.bfloat16

    # ---------------- attention ----------------
    def rope_cs(self, pos: torch.Tensor):
        fr = pos.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((fr, fr), dim=-1)
        return (emb.cos().to(self.dtype)[None, None],
                emb.sin().to(self.dtype)[None, None])

    @staticmethod
    def _rot(t: torch.Tensor) -> torch.Tensor:
        h = t.shape[-1] // 2
        return torch.cat((-t[..., h:], t[..., :h]), dim=-1)

    def attn(self, l: int, x: torch.Tensor, cache: dict, start_pos: int):
        s = self.spec
        T = x.shape[1]
        q = x @ self.wq[l].T
        k = x @ self.wk[l].T
        v = x @ self.wv[l].T
        if s.bias_qkv:
            q = q + self.bq[l]
            k = k + self.bk[l]
            v = v + self.bv[l]
        if s.clip_qkv is not None:
            q = q.clamp(-s.clip_qkv, s.clip_qkv)
            k = k.clamp(-s.clip_qkv, s.clip_qkv)
            v = v.clamp(-s.clip_qkv, s.clip_qkv)
        if s.qk_norm == "whole":
            # OLMoE style: RMSNorm over whole hidden vector BEFORE head view
            q = self._qk_rms(q, self.qn[l], s.eps)
            k = self._qk_rms(k, self.kn[l], s.eps)
        q = q.view(1, T, s.n_heads, s.head_dim)
        k = k.view(1, T, s.n_kv, s.head_dim)
        v = v.view(1, T, s.n_kv, s.head_dim).transpose(1, 2)
        if s.qk_norm == "per_head":
            q = self._qk_rms(q, self.qn[l], s.eps).transpose(1, 2)
            k = self._qk_rms(k, self.kn[l], s.eps).transpose(1, 2)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
        pos = torch.arange(start_pos, start_pos + T)
        cos, sin = self.rope_cs(pos)
        q = q * cos + self._rot(q) * sin
        k = k * cos + self._rot(k) * sin
        if l in cache:
            cache[l] = (torch.cat([cache[l][0], k], 2), torch.cat([cache[l][1], v], 2))
        else:
            cache[l] = (k, v)
        K, V = cache[l]
        rep = s.gqa_rep()
        if rep > 1:
            K = K.repeat_interleave(rep, dim=1)
            V = V.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, K, V, is_causal=(T > 1))
        o = o.transpose(1, 2).reshape(1, T, s.n_heads * s.head_dim)
        return o @ self.wo[l].T

    @staticmethod
    def _qk_rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        return n.to(x.dtype) * w

    def _whole_rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self._qk_rms(x, w)

    # ---------------- ffn ----------------
    def ffn_dense(self, l: int, h: torch.Tensor) -> torch.Tensor:
        p = f"model.layers.{l}.mlp"
        g = self.store.full(f"{p}.gate_proj.weight")
        u = self.store.full(f"{p}.up_proj.weight")
        dn = self.store.full(f"{p}.down_proj.weight")
        return (F.silu(h @ g.T) * (h @ u.T)) @ dn.T

    def route(self, l: int, h: torch.Tensor):
        s = self.spec
        logits = h @ self.gate[l].to(h.dtype).T
        probs = torch.softmax(logits.float(), dim=-1)[0]
        top = torch.topk(probs, s.top_k, dim=-1)
        w = top.values
        if s.norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
        return top.indices, w

    def ffn_moe_prefill(self, l: int, h: torch.Tensor, pool) -> torch.Tensor:
        """Union-aware batched prefill via fused Q4 GEMM (raw pools)."""
        from jouleai.native import kernel
        s = self.spec
        T = h.shape[1]
        idx, w = self.route(l, h)
        flat = idx.reshape(-1)
        uniq, inverse = torch.unique(flat, return_inverse=True)
        row_of = torch.arange(T).repeat_interleave(s.top_k)
        flat_w = w.reshape(-1)
        out = torch.zeros(1, T, s.d, dtype=h.dtype)
        X = h[0].float()
        m_g = s.intermediate

        def expert_block(j: int) -> tuple[torch.Tensor, torch.Tensor]:
            recs = pool.ensure(l, int(uniq[j].item()))
            Xd = X.to(h.dtype)
            if isinstance(recs[0], tuple):   # raw Q4 records -> fused kernel GEMM
                sc_g, pk_g, n_g = recs[0]
                sc_u, pk_u, n_u = recs[1]
                sc_d, pk_d, n_d = recs[2]
                G = torch.empty(T, m_g, dtype=torch.float32)
                kernel.q4_gemm(G, X, pk_g, sc_g, m_g, s.d)
                U = torch.empty(T, m_g, dtype=torch.float32)
                kernel.q4_gemm(U, X, pk_u, sc_u, m_g, s.d)
                act = F.silu(G) * U
                D = torch.empty(T, s.d, dtype=torch.float32)
                kernel.q4_gemm(D, act, pk_d, sc_d, s.d, m_g)
            else:                            # bf16 tensors -> torch batched
                g, u, dn = recs
                act = F.silu(Xd @ g.T) * (Xd @ u.T)
                D = (act @ dn.T).float()
            sel = (inverse == j).nonzero(as_tuple=True)[0]
            rows = row_of[sel]
            y = flat_w[sel].float().unsqueeze(1) * D[rows]
            return rows, y

        ex = getattr(self, "_executor", None)
        n_uniq = len(uniq)
        if ex is not None and n_uniq > 1:
            results = list(ex.map(expert_block, range(n_uniq)))
        else:
            results = [expert_block(j) for j in range(n_uniq)]
        for rows, y in results:
            out[0].index_add_(0, rows, y.to(h.dtype))
        return out

    def ffn_moe_decode(self, l: int, h: torch.Tensor, pool) -> torch.Tensor:
        """Decode FFN via AVX2 batched q4_gemm (expert_ffn.dll, one call/expert).

        Each expert's gate/up/down is one fused AVX2 q4_gemm (no threads, no
        Python dispatch). Profiled: ~2x faster than the C expert_job thread
        path, ~4x vs the original per-op q4_gemv loop.
        """
        from jouleai.native import kernel
        s = self.spec
        idx, w = self.route(l, h)
        x = h[0, -1].float()                       # [d]
        out = torch.zeros(s.d, dtype=torch.float32)
        for i, e in enumerate(idx[0].tolist()):
            recs = pool.ensure(l, int(e))
            if isinstance(recs[0], tuple):         # raw Q4 records -> fused kernel
                sc_g, pk_g, n_g = recs[0]
                sc_u, pk_u, n_u = recs[1]
                sc_d, pk_d, n_d = recs[2]
                # AVX2 q4_gemm with T=1 == a fast GEMV (fused dequant)
                G = torch.empty(1, n_g // s.d, dtype=torch.float32)
                kernel.q4_gemm(G, x.unsqueeze(0), pk_g, sc_g, n_g // s.d, s.d)
                U = torch.empty(1, n_u // s.d, dtype=torch.float32)
                kernel.q4_gemm(U, x.unsqueeze(0), pk_u, sc_u, n_u // s.d, s.d)
                act = F.silu(G) * U
                D = torch.empty(1, s.d, dtype=torch.float32)
                kernel.q4_gemm(D, act, pk_d, sc_d, s.d, s.intermediate)
            else:                                  # bf16 tensors -> torch matmul
                g, u, dn = recs
                xb = x.to(g.dtype)
                act = F.silu(xb @ g.T) * (xb @ u.T)
                D = (act @ dn.T).float()
            out += w[0, i].float() * D[0]
        return out.unsqueeze(0).unsqueeze(0).to(h.dtype)

    # ---------------- forward / generate ----------------
    @staticmethod
    def _kernel_gemv(x, pk, sc, m, d):
        from jouleai.native import kernel
        return kernel.q4_gemv(x[0], pk, sc, m, d)

    def forward_batch(self, input_ids: torch.Tensor, caches: list[dict],
                      start_pos, pool=None) -> torch.Tensor:
        # start_pos: int (same for all) OR list[int] (per-sequence)
        positions = start_pos if isinstance(start_pos, (list, tuple))             else [start_pos] * input_ids.shape[0]
        """Batched decode: B sequences' current tokens [B, 1], shared weights.

        Each sequence has its own KV cache; attention is per-sequence (B,
        T=1) but all weight reads are shared (torch matmul batches B rows).
        Returns logits [B, V].
        """
        s = self.spec
        B = input_ids.shape[0]
        x = self.embed[input_ids[:, 0]]                      # [B, d]

        def rms(v, w):
            return (v.float() * torch.rsqrt(v.float().pow(2).mean(-1, keepdim=True)
                                            + s.eps)).to(v.dtype) * w
        for l in range(s.n_layers):
            h = rms(x, self.norm1[l])
            # per-seq attention: each sequence has its own KV/position
            outs_a = []
            for b in range(B):
                outs_a.append(self.attn(l, h[b:b+1].unsqueeze(0), caches[b],
                                        positions[b])[0, 0])
            x = x + torch.stack(outs_a)
            h = rms(x, self.norm2[l])
            if s.moe:
                outs = []
                for b in range(B):
                    outs.append(self.ffn_moe_decode(l, h[b:b+1].unsqueeze(0),
                                                    pool)[0, 0])
                x = x + torch.stack(outs)   # [B, d]
            else:
                x = x + self.ffn_dense(l, h)
        return rms(x, self.final_norm) @ self.lm_head.T

    def _ffn_moe_batch(self, l: int, h: torch.Tensor, pool) -> torch.Tensor:
        """Union-aware batched expert FFN: B sequences share weight reads.

        Collects unique experts across all B sequences, gathers each once,
        and computes the gated activation for the B rows that selected it.
        """
        s = self.spec
        B = h.shape[0]
        # route each sequence
        idxs, ws = [], []
        for b in range(B):
            idx, w = self.route(l, h[b:b+1])
            idxs.append(idx[0])
            ws.append(w[0])
        # unique experts across batch
        all_idx = torch.cat(idxs)
        uniq, inverse = torch.unique(all_idx, return_inverse=True)
        out = torch.zeros(B, s.d, dtype=h.dtype)
        for j, e in enumerate(uniq.tolist()):
            recs = pool.ensure(l, int(e))
            sc_g, pk_g, n_g = recs[0]
            sc_u, pk_u, n_u = recs[1]
            sc_d, pk_d, n_d = recs[2]
            # which (b, slot) selected this expert
            for b in range(B):
                mask = (idxs[b] == e)
                if not mask.any():
                    continue
                xb = h[b:b+1].float()
                # gated activation for this expert
                g = self._kernel_gemv(xb, pk_g, sc_g, n_g // s.d, s.d)
                u = self._kernel_gemv(xb, pk_u, sc_u, n_u // s.d, s.d)
                act = F.silu(g) * u
                w_sel = ws[b][mask].float()
                y = self._kernel_gemv(act, pk_d, sc_d, s.d, s.intermediate)
                out[b] += (w_sel * y).sum(0)
        return out

    def _attn_batch(self, l: int, x: torch.Tensor, caches: list[dict],
                    start_pos: int) -> torch.Tensor:
        """Per-sequence attention with batched weight projection [B, d]."""
        s = self.spec
        B = x.shape[0]
        # batched projections: [B, d] @ W^T -> [B, n_heads, hd]
        q = (x @ self.wq[l].T).view(B, s.n_heads, s.head_dim)
        k = (x @ self.wk[l].T).view(B, s.n_kv, s.head_dim)
        v = (x @ self.wv[l].T).view(B, s.n_kv, s.head_dim)
        if s.qk_norm == "per_head":
            q = self._qk_rms(q, self.qn[l], s.eps)
            k = self._qk_rms(k, self.kn[l], s.eps)
        # RoPE (all sequences at same pos)
        cos, sin = self.rope_cs(torch.tensor([start_pos]))
        q = q * cos[0, 0, 0] + self._rot(q) * sin[0, 0, 0]
        k = k * cos[0, 0, 0] + self._rot(k) * sin[0, 0, 0]
        outs = []
        for b in range(B):
            # attn() stores KV as [1, n_kv, T, hd] (transposed, dim2=time)
            kb = k[b].unsqueeze(0)                 # [1, n_kv, hd] -> [1,1,n_kv,hd]?
            # k[b] is [n_kv, hd]; cache format is [1, n_kv, T, hd]
            kb = k[b].unsqueeze(0).unsqueeze(0).transpose(1, 2)  # [1,4,1,128]
            vb = v[b].unsqueeze(0).unsqueeze(0).transpose(1, 2)
            if l in caches[b]:
                caches[b][l] = (torch.cat([caches[b][l][0], kb], 2),
                                torch.cat([caches[b][l][1], vb], 2))
            else:
                caches[b][l] = (kb, vb)
            K, V = caches[b][l]                    # [1, n_kv, T, hd]
            rep = s.gqa_rep()
            if rep > 1:
                K = K.repeat_interleave(rep, dim=1)  # [1, H, T, hd]
                V = V.repeat_interleave(rep, dim=1)
            qb = q[b].unsqueeze(0).unsqueeze(0)      # [1,1,H,hd] -> q [B,H,hd]
            # qb should be [1, H, 1, hd] for SDPA: q[b] is [H,hd]
            qb = q[b].unsqueeze(0).unsqueeze(0)      # [1,1,H,hd]
            o = F.scaled_dot_product_attention(qb, K.transpose(1, 2),
                                               V.transpose(1, 2), is_causal=True)
            outs.append(o[0, 0].reshape(-1))
        attn_out = torch.stack(outs)                # [B, n_heads*hd]
        return attn_out @ self.wo[l].T              # [B, d]

    def forward(self, ids, cache: dict, start_pos: int, pool=None) -> torch.Tensor:
        s = self.spec
        input_ids = ids["input_ids"] if not torch.is_tensor(ids) else ids
        x = self.embed[input_ids[0]].unsqueeze(0)

        def rms(v, w):
            return (v.float() * torch.rsqrt(v.float().pow(2).mean(-1, keepdim=True)
                                            + s.eps)).to(v.dtype) * w
        for l in range(s.n_layers):
            h = rms(x, self.norm1[l])
            x = x + self.attn(l, h, cache, start_pos)
            h = rms(x, self.norm2[l])
            if s.moe:
                x = x + (self.ffn_moe_prefill(l, h, pool) if h.shape[1] > 1
                         else self.ffn_moe_decode(l, h, pool))
            else:
                x = x + self.ffn_dense(l, h)
        return rms(x, self.final_norm) @ self.lm_head.T

    def generate(self, ids, max_new: int, pool, eos_id: int | None = None):
        cache: dict = {}
        t0 = time.perf_counter()
        logits = self.forward(ids, cache, 0, pool)
        p_len = (ids["input_ids"] if not torch.is_tensor(ids) else ids).shape[1]
        out = [int(logits[0, -1].argmax())]
        for _ in range(max_new - 1):
            if eos_id is not None and out[-1] == eos_id:
                break
            step = torch.tensor([[out[-1]]])
            logits = self.forward(step, cache, p_len + len(out) - 1, pool)
            out.append(int(logits[0, -1].argmax()))
        wall = time.perf_counter() - t0
        return {"ids": out, "tok_s": round((len(out) - 1) / max(wall - 0, 1e-9), 2),
                "wall_s": wall}
