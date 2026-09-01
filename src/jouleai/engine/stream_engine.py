"""StreamEngine — Qwen2-style forward pass backed by SenseWeightStore.

Database-style inference:
  - fixed weights (embed, attention, norms, lm_head) are resident in RAM
  - FFN neurons are ADDRESSED per query: prefill computes full FFN (and captures
    the last-position activation per layer -> TopMass masks), then a per-query
    FFNRowPool gathers only the kept neuron rows from the mmap-backed store.
  - the pool is released after the query -> RAM returns to the fixed set.

This is the "load only what you need, release after use" engine. Numerics
follow HF Qwen2 (RMSNorm in fp32, RoPE theta from config, GQA, causal SDPA).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from jouleai.routing.mask_policy import MaskPolicy, TopMassPolicy
from jouleai.storage.weight_store import SenseWeightStore


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    v = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (v.to(x.dtype) * w)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class FFNRowPool:
    """Per-query resident subset of FFN neuron rows (released after the query)."""

    def __init__(self):
        self.gate: list[torch.Tensor] = []
        self.up: list[torch.Tensor] = []
        self.down: list[torch.Tensor] = []  # [k, d_model] each: kept neuron rows
        self.idx: list[torch.Tensor | None] = []  # sorted neuron indices per layer
        self.masks: list[torch.Tensor] = []
        self.touched_bytes = 0


class StreamEngine:
    def __init__(self, model_dir: str | Path,
                 down_t_dir: str | Path | None = None,
                 policy: MaskPolicy | None = None):
        self.dir = Path(model_dir)
        self.cfg = json.loads((self.dir / "config.json").read_text())
        self.store = SenseWeightStore(self.dir)
        self.down_t = SenseWeightStore(down_t_dir) if down_t_dir else None
        self.policy = policy or TopMassPolicy(0.9)

        c = self.cfg
        self.n_layers = c["num_hidden_layers"]
        self.d = c["hidden_size"]
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.hd = self.d // self.n_heads
        self.eps = c["rms_norm_eps"]
        self.theta = c.get("rope_theta", 1e6)
        self.tied = c.get("tie_word_embeddings", False)
        self.dtype = torch.bfloat16

        self._load_fixed()
        self.touched_bytes = 0        # cumulative FFN pool bytes gathered
        self.prefill_touched_bytes = 0  # cumulative full-FFN bytes touched in prefill
        self.collect: list | None = None  # calibration: per-layer [(h, act)] stasher
        self.probes = None                # ProbeBank when running mask_source="probe"

    # ---------------------------------------------------------------- fixed
    def _load_fixed(self) -> None:
        s, L = self.store, self.n_layers
        self.embed = s.full("model.embed_tokens.weight")          # [V, d]
        self.lm_head = self.embed if self.tied else s.full("lm_head.weight")
        self.final_norm = s.full("model.norm.weight")
        self.wq, self.bq, self.wk, self.bk = [], [], [], []
        self.wv, self.bv, self.wo = [], [], []
        self.norm1, self.norm2 = [], []
        for l in range(L):
            p = f"model.layers.{l}"
            self.wq.append(s.full(f"{p}.self_attn.q_proj.weight"))
            self.wk.append(s.full(f"{p}.self_attn.k_proj.weight"))
            self.wv.append(s.full(f"{p}.self_attn.v_proj.weight"))
            self.wo.append(s.full(f"{p}.self_attn.o_proj.weight"))
            self.bq.append(s.full(f"{p}.self_attn.q_proj.bias"))
            self.bk.append(s.full(f"{p}.self_attn.k_proj.bias"))
            self.bv.append(s.full(f"{p}.self_attn.v_proj.bias"))
            self.norm1.append(s.full(f"{p}.input_layernorm.weight"))
            self.norm2.append(s.full(f"{p}.post_attention_layernorm.weight"))
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.inv_freq = inv  # [hd/2]

    def rope_cs(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fr = positions.float()[:, None] * self.inv_freq[None, :]  # [T, hd/2]
        emb = torch.cat((fr, fr), dim=-1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    # ---------------------------------------------------------------- ffn
    def _ffn_full(self, l: int, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full FFN (prefill). Returns (out [.., d], act_last [d_ff])."""
        p = f"model.layers.{l}.mlp"
        g = self.store.full(f"{p}.gate_proj.weight")
        u = self.store.full(f"{p}.up_proj.weight")
        dn = self.store.full(f"{p}.down_proj.weight")
        self.prefill_touched_bytes += self.store.bytes_of(f"{p}.gate_proj.weight") \
            + self.store.bytes_of(f"{p}.up_proj.weight") \
            + self.store.bytes_of(f"{p}.down_proj.weight")
        act = F.silu(h @ g.T) * (h @ u.T)             # [1, T, d_ff]
        if self.collect is not None:
            self.collect.append((h[0].detach(), act[0].detach()))
        out = act @ dn.T                              # [1, T, d]
        return out, act[0, -1].detach().clone()       # last position [d_ff]

    def _ffn_refresh(self, l: int, h: torch.Tensor, pool: FFNRowPool,
                     capture: list) -> torch.Tensor:
        """Refresh-step FFN: exact activations via full gate/up (for the mask),
        pooled down_proj for the output. Cost ≈ 2/3 of a full FFN step."""
        p = f"model.layers.{l}.mlp"
        g_full = self.store.full(f"{p}.gate_proj.weight")
        u_full = self.store.full(f"{p}.up_proj.weight")
        self.prefill_touched_bytes += self.store.bytes_of(f"{p}.gate_proj.weight") \
            + self.store.bytes_of(f"{p}.up_proj.weight")
        act = F.silu(h @ g_full.T) * (h @ u_full.T)      # [1, T, d_ff] exact
        capture.append(act[0, -1].detach().clone())
        act_pooled = act[0][:, pool.idx[l]].unsqueeze(0)  # [1, k] aligned to pool
        return act_pooled @ pool.down[l]                  # [1, T, d]

    def _ffn_pool(self, l: int, h: torch.Tensor, pool: FFNRowPool) -> torch.Tensor:
        """Neuron-selective FFN (decode): only pooled rows contribute."""
        g, u, dn = pool.gate[l], pool.up[l], pool.down[l]   # [k,d],[k,d],[k,d]
        act = F.silu(h @ g.T) * (h @ u.T)             # [1, k]
        return act @ dn                               # [1, d]

    def _update_pool_layer(self, l: int, idx_new: torch.Tensor,
                           pool: FFNRowPool, cap_frac: float = 0.70) -> None:
        """Update layer l pool rows toward the target mask.

        Union-with-cap policy: keep still-needed rows and rows added earlier
        (avoids churn), gather only missing rows; drop surplus rows only when
        the pool exceeds cap_frac of d_ff.
        """
        old = pool.idx[l]
        d_ff = self.store.shape_of(
            f"model.layers.{l}.mlp.gate_proj.weight")[0]
        union = torch.cat([old, idx_new[~torch.isin(idx_new, old)]])
        union = union.unique()
        if union.numel() > int(cap_frac * d_ff):
            # over budget: fall back to tracking the target mask exactly
            union = idx_new.unique()
        missing = union[~torch.isin(union, old)]
        if missing.numel() == 0 and union.numel() == old.numel():
            return
        keep_mask = torch.isin(old, union)
        sg = pool.gate[l][keep_mask]
        su = pool.up[l][keep_mask]
        sd = pool.down[l][keep_mask]
        s_idx = old[keep_mask]
        if missing.numel():
            g, u, dn = self._gather_rows(l, missing)
            sg = torch.cat([sg, g]); su = torch.cat([su, u]); sd = torch.cat([sd, dn])
            s_idx = torch.cat([s_idx, missing])
            added = missing.numel() * 3 * self.d * 2
            pool.touched_bytes += added
            self.touched_bytes += added
            pool.gathered_bytes = getattr(pool, "gathered_bytes", 0) + added
        order = torch.argsort(s_idx)
        pool.gate[l], pool.up[l], pool.down[l] = sg[order], su[order], sd[order]
        pool.idx[l] = s_idx[order]

    def _ffn_probe(self, l: int, h: torch.Tensor, pool: FFNRowPool) -> torch.Tensor:
        """Probe-driven decode step: predict activations from h, update the
        layer's pool to the predicted mask (delta gather), then pooled FFN."""
        act_pred = self.probes.predict_act(l, h)[0, -1]
        idx_new = self.policy.mask(act_pred.abs()).nonzero(as_tuple=True)[0]
        idx_new, _ = idx_new.sort()
        self._update_pool_layer(l, idx_new, pool)
        return self._ffn_pool(l, h, pool)

    def build_pool(self, masks: list[torch.Tensor]) -> FFNRowPool:
        pool = FFNRowPool()
        for l, m in enumerate(masks):
            idx = m.nonzero(as_tuple=True)[0]
            p = f"model.layers.{l}.mlp"
            g = self.store.rows(f"{p}.gate_proj.weight", idx)
            u = self.store.rows(f"{p}.up_proj.weight", idx)
            if self.down_t is not None:
                dn = self.down_t.rows(f"{p}.down_proj_t.weight", idx)   # [k, d]
            else:
                dn = self.store.cols(f"{p}.down_proj.weight", idx).T    # [k, d]
            pool.gate.append(g.to(self.dtype))
            pool.up.append(u.to(self.dtype))
            pool.down.append(dn.to(self.dtype))
            pool.idx.append(idx)
            pool.masks.append(m)
            pool.touched_bytes += idx.numel() * 3 * self.d * 2
        self.touched_bytes += pool.touched_bytes
        return pool

    def _gather_rows(self, l: int, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = f"model.layers.{l}.mlp"
        g = self.store.rows(f"{p}.gate_proj.weight", idx).to(self.dtype)
        u = self.store.rows(f"{p}.up_proj.weight", idx).to(self.dtype)
        if self.down_t is not None:
            dn = self.down_t.rows(f"{p}.down_proj_t.weight", idx).to(self.dtype)
        else:
            dn = self.store.cols(f"{p}.down_proj.weight", idx).T.to(self.dtype)
        return g, u, dn

    def refresh_pool(self, pool: FFNRowPool, masks: list[torch.Tensor]) -> dict:
        """Adaptive predictor step: exact masks from a full-FFN refresh token;
        gather only the delta rows and merge into the pool (union grows)."""
        added = 0
        t0 = time.perf_counter()
        for l, m in enumerate(masks):
            idx = m.nonzero(as_tuple=True)[0]
            old = pool.idx[l]
            is_new = ~torch.isin(idx, old)
            missing = idx[is_new]
            if missing.numel() == 0:
                pool.masks[l] = m
                continue
            g, u, dn = self._gather_rows(l, missing)
            merged_idx = torch.cat([old, missing])
            order = torch.argsort(merged_idx)
            pool.gate[l] = torch.cat([pool.gate[l], g])[order]
            pool.up[l] = torch.cat([pool.up[l], u])[order]
            pool.down[l] = torch.cat([pool.down[l], dn])[order]
            pool.idx[l] = merged_idx[order]
            pool.masks[l] = m
            added += missing.numel()
            pool.touched_bytes += missing.numel() * 3 * self.d * 2
            self.touched_bytes += missing.numel() * 3 * self.d * 2
        return {"added_rows": added, "refresh_s": time.perf_counter() - t0}

    @staticmethod
    def release_pool(pool: FFNRowPool | None) -> None:
        if pool is None:
            return
        pool.gate.clear(); pool.up.clear(); pool.down.clear()
        pool.masks.clear(); pool.idx.clear()

    # ---------------------------------------------------------------- attention
    def _attn(self, l: int, x: torch.Tensor, cache: dict, start_pos: int) -> torch.Tensor:
        T = x.shape[1]
        q = (x @ self.wq[l].T + self.bq[l]).view(1, T, self.n_heads, self.hd).transpose(1, 2)
        k = (x @ self.wk[l].T + self.bk[l]).view(1, T, self.n_kv, self.hd).transpose(1, 2)
        v = (x @ self.wv[l].T + self.bv[l]).view(1, T, self.n_kv, self.hd).transpose(1, 2)
        pos = torch.arange(start_pos, start_pos + T)
        cos, sin = self.rope_cs(pos)              # [T, hd]
        cos = cos[None, None, :, :]               # [1, 1, T, hd] for q [B, H, T, hd]
        sin = sin[None, None, :, :]
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        if l in cache:
            cache[l] = (torch.cat([cache[l][0], k], dim=2),
                        torch.cat([cache[l][1], v], dim=2))
        else:
            cache[l] = (k, v)
        K, V = cache[l]
        rep = self.n_heads // self.n_kv
        if rep > 1:
            K = K.repeat_interleave(rep, dim=1)
            V = V.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, K, V, is_causal=(T > 1))
        o = o.transpose(1, 2).reshape(1, T, self.d)
        return o @ self.wo[l].T

    # ---------------------------------------------------------------- forward
    def forward(self, ids: torch.Tensor, cache: dict, start_pos: int,
                pool: FFNRowPool | None, capture: list | None,
                refresh: bool = False, probe: bool = False) -> torch.Tensor:
        """One forward over input_ids [1, T]; returns logits [1, T, V].

        refresh=True (decode steps): per layer compute exact activations from
        full gate/up (mask source) but pooled down_proj (output).
        probe=True (decode steps): per-layer masks come from ProbeBank
        predictions; the pool tracks the predicted mask via delta gathers.
        """
        input_ids = ids["input_ids"] if not torch.is_tensor(ids) else ids
        x = self.embed[input_ids[0]].unsqueeze(0)  # [1, T, d] bf16
        T = input_ids.shape[1]
        for l in range(self.n_layers):
            h = rms_norm(x, self.norm1[l], self.eps)
            x = x + self._attn(l, h, cache, start_pos)
            h = rms_norm(x, self.norm2[l], self.eps)
            if pool is not None and probe:
                x = x + self._ffn_probe(l, h, pool)
            elif pool is not None and refresh:
                cap_l: list = []
                x = x + self._ffn_refresh(l, h, pool, cap_l)
                if capture is not None:
                    capture.append(cap_l[0])
            elif pool is not None:
                x = x + self._ffn_pool(l, h, pool)
            else:
                out, act_last = self._ffn_full(l, h)
                if capture is not None:
                    capture.append(act_last)
                x = x + out
        return rms_norm(x, self.final_norm, self.eps) @ self.lm_head.T

    # ---------------------------------------------------------------- generate
    def generate(self, ids, max_new: int, mass: float = 0.9,
                 keep: float = 1.0, verify_k: int = 0,
                 eos_id: int | None = None,
                 refresh_every: int = 0,
                 mask_source: str = "static", probes=None) -> dict:
        """Greedy generate with database-style FFN pooling + optional verify gate.

        keep < 1.0: pool only neurons needed to cover `mass` activation mass.
        mask_source: "static" (prefill-captured masks), "probe" (per-token masks
        predicted by ProbeBank — pool tracks them via delta gathers), or use
        refresh_every>0 for exact-scan refreshes.
        verify_k > 0: margin-tolerant greedy-exactness gate (full recompute of
        the first k tokens; near-tie flips accepted, large-margin flips serve
        the full-keep answer).
        """
        from jouleai.routing.probe_bank import ProbeBank  # local: avoid cycle
        if mask_source == "probe":
            assert isinstance(probes, ProbeBank) and len(probes) == self.n_layers
            self.probes = probes
        cache: dict = {}
        capture: list = []
        logits = self.forward(ids, cache, 0, None, capture)
        prompt_len = ids["input_ids"].shape[1] if not torch.is_tensor(ids) else ids.shape[1]

        masks = [self.policy.mask(a) for a in capture]
        keep_ratios = [float((m.sum() / m.numel()).item()) for m in masks]

        use_pool = keep < 1.0
        pool = self.build_pool(masks) if use_pool else None
        if pool is not None:
            pool.gathered_bytes = pool.touched_bytes

        out_ids: list[int] = []
        tok = int(logits[0, -1].argmax())
        out_ids.append(tok)
        prefill_mb = self.prefill_touched_bytes / 1048576
        refresh_stats = {"n_refresh": 0, "added_rows": 0, "refresh_s": 0.0}
        scan_bytes0 = self.prefill_touched_bytes

        t0 = time.perf_counter()
        for step_i in range(max_new - 1):
            if eos_id is not None and out_ids[-1] == eos_id:
                break
            step = torch.tensor([[out_ids[-1]]])
            pos = prompt_len + len(out_ids) - 1
            if use_pool and mask_source == "probe":
                # probe refreshes masks every refresh_every tokens (cadence);
                # other tokens decode with the current pooled rows (no churn)
                if refresh_every and step_i % refresh_every == 0:
                    logits = self.forward(step, cache, pos, pool, None, probe=True)
                    keep_ratios = [float((m.sum() / m.numel()).item())
                                   for m in pool.masks]
                else:
                    logits = self.forward(step, cache, pos, pool, None)
            elif use_pool and refresh_every and step_i % refresh_every == 0:
                # exact-mask refresh step (gate/up full, output pooled);
                # includes step 0 so the first decode token is also refreshed
                cap2: list = []
                logits = self.forward(step, cache, pos, pool, cap2, refresh=True)
                new_masks = [self.policy.mask(a) for a in cap2]
                st = self.refresh_pool(pool, new_masks)
                refresh_stats["n_refresh"] += 1
                refresh_stats["added_rows"] += st["added_rows"]
                refresh_stats["refresh_s"] += st["refresh_s"]
                keep_ratios = [float((m.sum() / m.numel()).item()) for m in new_masks]
            else:
                logits = self.forward(step, cache, pos, pool, None)
            out_ids.append(int(logits[0, -1].argmax()))
        decode_s = time.perf_counter() - t0
        refresh_stats["refresh_scan_mb"] = (self.prefill_touched_bytes - scan_bytes0) / 1048576

        answer_ids = out_ids
        gate = {"used": False}
        if use_pool and verify_k > 0:
            gate = self._verify(ids, out_ids, verify_k, eos_id)
            if gate["fell_back"]:
                answer_ids = gate["full_ids"]

        self.release_pool(pool)
        if mask_source == "probe":
            self.probes = None
        import gc
        gc.collect()
        return {
            "ids": answer_ids,
            "keep_ratios": keep_ratios,
            "keep_mean": sum(keep_ratios) / len(keep_ratios),
            "pool_touched_mb": pool.touched_bytes / 1048576 if pool else 0.0,
            "pool_gathered_mb": (getattr(pool, "gathered_bytes", 0) or 0) / 1048576,
            "prefill_touched_mb": prefill_mb,
            "decode_s_per_tok": decode_s / max(len(out_ids) - 1, 1),
            "refresh": refresh_stats,
            "gate": gate,
        }

    def _verify(self, ids, masked_ids: list[int], k: int,
                eos_id: int | None, margin_eps: float = 0.5) -> dict:
        """Recompute first k tokens with full FFN; margin-tolerant acceptance.

        At the first divergent position, if the masked token's logit under the
        FULL model is within `margin_eps` of the full top-1 (near-tie flip),
        the masked path is accepted; a large-margin flip falls back to full.
        """
        cache: dict = {}
        logits = self.forward(ids, cache, 0, None, None)
        p_len = ids["input_ids"].shape[1] if not torch.is_tensor(ids) else ids.shape[1]
        full_ids = [int(logits[0, -1].argmax())]
        step_logits = [logits[0, -1].float().clone()]
        for _ in range(k - 1):
            step = torch.tensor([[full_ids[-1]]])
            logits = self.forward(step, cache, p_len + len(full_ids) - 1,
                                  None, None)
            full_ids.append(int(logits[0, -1].argmax()))
            step_logits.append(logits[0, -1].float().clone())

        for t in range(min(k, len(masked_ids))):
            if masked_ids[t] == full_ids[t]:
                continue
            fl = step_logits[t]
            margin = (fl[full_ids[t]] - fl[masked_ids[t]]).item()
            if margin <= margin_eps:
                return {"used": True, "fell_back": False,
                        "full_ids": full_ids,
                        "note": f"near-tie flip accepted at token {t} "
                                f"(margin {margin:.2f} <= {margin_eps})"}
            # large-margin flip -> full greedy continuation (lossless fallback)
            for _t in range(len(masked_ids) - k):
                step = torch.tensor([[full_ids[-1]]])
                logits2 = self.forward(step, cache, p_len + len(full_ids) - 1,
                                       None, None)
                nxt = int(logits2[0, -1].argmax())
                if eos_id is not None and nxt == eos_id:
                    break
                full_ids.append(nxt)
            return {"used": True, "fell_back": True, "full_ids": full_ids,
                    "note": f"large-margin flip at token {t} "
                            f"(margin {margin:.2f}); served full-keep answer"}
        return {"used": True, "fell_back": False, "full_ids": full_ids,
                "note": f"first {k} tokens verified greedy-exact"}
