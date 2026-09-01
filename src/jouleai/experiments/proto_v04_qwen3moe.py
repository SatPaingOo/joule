"""Qwen3-family validation: dense path correctness + frontier MoE streaming.

Modes:
  dense-validate  local Qwen3-8B (dense): manual forward vs HF (logits + greedy)
  moe-stream      Qwen3-30B-A3B (61 GB, CANNOT fully load on this machine):
                  sparse expert streaming at a RAM budget; metrics + self-
                  consistency across budgets (full-serve baseline impossible
                  here — that is the point).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.storage.expert_store import ExpertLRUPool  # noqa: E402
from jouleai.storage.weight_store import SenseWeightStore  # noqa: E402

QUERIES = {
    "easy": "What is the capital of France? Answer in one sentence.",
    "medium": "Explain photosynthesis in simple terms, in about 3 sentences.",
    "hard": (
        "A train travels 120 km in 90 minutes. Its speed then halves. "
        "How long will it take to travel another 60 km? Reason step by step, "
        "then give the final answer."
    ),
}


class Qwen3Streamer:
    """Config-driven Qwen3 / Qwen3-MoE forward with expert streaming."""

    def __init__(self, model_dir: Path):
        self.cfg = json.loads((model_dir / "config.json").read_text())
        self.dir = model_dir
        self.store = SenseWeightStore(model_dir)
        c = self.cfg
        self.is_moe = "num_experts" in c
        self.n_layers = c["num_hidden_layers"]
        self.d = c["hidden_size"]
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.hd = c.get("head_dim", self.d // self.n_heads)
        self.eps = c["rms_norm_eps"]
        self.theta = c["rope_theta"]
        self.tied = c.get("tie_word_embeddings", False)
        self.embed = self.store.full("model.embed_tokens.weight")
        self.lm_head = self.embed if self.tied else self.store.full("lm_head.weight")
        self.final_norm = self.store.full("model.norm.weight")
        self.wq, self.wk, self.wv, self.wo = [], [], [], []
        self.qn, self.kn = [], []
        self.norm1, self.norm2 = [], []
        self.gate = []
        for l in range(self.n_layers):
            p = f"model.layers.{l}"
            self.wq.append(self.store.full(f"{p}.self_attn.q_proj.weight"))
            self.wk.append(self.store.full(f"{p}.self_attn.k_proj.weight"))
            self.wv.append(self.store.full(f"{p}.self_attn.v_proj.weight"))
            self.wo.append(self.store.full(f"{p}.self_attn.o_proj.weight"))
            self.qn.append(self.store.full(f"{p}.self_attn.q_norm.weight"))
            self.kn.append(self.store.full(f"{p}.self_attn.k_norm.weight"))
            self.norm1.append(self.store.full(f"{p}.input_layernorm.weight"))
            self.norm2.append(self.store.full(f"{p}.post_attention_layernorm.weight"))
            if self.is_moe:
                self.gate.append(self.store.full(f"{p}.mlp.gate.weight"))
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.inv_freq = inv

    def rope_cs(self, pos: torch.Tensor):
        fr = pos.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((fr, fr), dim=-1)
        return (emb.cos().to(torch.bfloat16)[None, None],
                emb.sin().to(torch.bfloat16)[None, None])

    @staticmethod
    def _rot(t: torch.Tensor) -> torch.Tensor:
        h = t.shape[-1] // 2
        return torch.cat((-t[..., h:], t[..., :h]), dim=-1)

    @staticmethod
    def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                        + eps)).to(x.dtype) * w

    def attn(self, l: int, x: torch.Tensor, cache: dict, start_pos: int):
        T = x.shape[1]
        q = (x @ self.wq[l].T).view(1, T, -1, self.hd)
        k = (x @ self.wk[l].T).view(1, T, -1, self.hd)
        v = (x @ self.wv[l].T).view(1, T, -1, self.hd).transpose(1, 2)
        q = (self._qk_rms(q, self.qn[l])).transpose(1, 2)
        k = (self._qk_rms(k, self.kn[l])).transpose(1, 2)
        pos = torch.arange(start_pos, start_pos + T)
        cos, sin = self.rope_cs(pos)
        q = q * cos + self._rot(q) * sin
        k = k * cos + self._rot(k) * sin
        if l in cache:
            cache[l] = (torch.cat([cache[l][0], k], 2), torch.cat([cache[l][1], v], 2))
        else:
            cache[l] = (k, v)
        K, V = cache[l]
        rep = self.n_heads // self.n_kv
        if rep > 1:
            K = K.repeat_interleave(rep, dim=1)
            V = V.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, K, V, is_causal=(T > 1))
        o = o.transpose(1, 2).reshape(1, T, self.n_heads * self.hd)
        return o @ self.wo[l].T

    def _qk_rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Per-head RMSNorm over head_dim (Qwen3 style), x [1, T, H, hd]."""
        n = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (n.to(x.dtype) * w)

    def route(self, l: int, h: torch.Tensor):
        logits = h @ self.gate[l].to(h.dtype).T
        probs = torch.softmax(logits.float(), dim=-1)[0]
        top = torch.topk(probs, self.cfg["num_experts_per_tok"], dim=-1)
        w = top.values
        if self.cfg.get("norm_topk_prob", False):
            w = w / w.sum(dim=-1, keepdim=True)
        return top.indices, w

    def ffn_moe(self, l: int, h: torch.Tensor, pool) -> torch.Tensor:
        """Dispatch: decode (T==1) uses the stacked fast path; prefill uses the
        union-aware batched path (correct for multi-position routing)."""
        if h.shape[1] == 1:
            return self._ffn_decode(l, h, pool)
        return self._ffn_prefill(l, h, pool)

    def _ffn_decode(self, l: int, h: torch.Tensor, pool) -> torch.Tensor:
        idx, w = self.route(l, h)                       # [1, k]
        experts = idx[0].tolist()
        first = pool.ensure(l, experts[0])
        if isinstance(first[0], tuple):
            return self._ffn_decode_native(l, h, pool, experts, w)
        tensors = [first] + [pool.ensure(l, e) for e in experts[1:]]
        g = torch.cat([t[0] for t in tensors])          # [k*m, d]
        u = torch.cat([t[1] for t in tensors])
        dn = torch.stack([t[2] for t in tensors])       # [k, m, d]
        x = h[0, -1]                                    # [d]
        act = F.silu(x @ g.T) * (x @ u.T)               # [k*m]
        m = act.shape[0] // len(tensors)
        y = torch.bmm(act.view(len(tensors), 1, m),
                      dn.transpose(1, 2)).squeeze(1)    # [k, d]
        out = (w[0].to(y.dtype).unsqueeze(1) * y).sum(0, keepdim=True)
        return out.unsqueeze(0)                         # [1, 1, d]

    def _ffn_decode_native(self, l: int, h: torch.Tensor, pool,
                           experts: list[int], w: torch.Tensor) -> torch.Tensor:
        """Raw-Q4 records + native kernel: fused dequant-GEMV, threaded."""
        nm = getattr(self, "_native_moe", None)
        if nm is not None:
            return nm.ffn(self, l, h, pool, experts, w,
                          executor=getattr(self, "_executor", None))
        from jouleai.native import kernel
        if getattr(self, "_executor", None) is None:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=8)
        ex = self._executor
        x = h[0, -1].float()                            # [d] fp32
        recs = [pool.ensure(l, e) for e in experts]     # [(sc,pk,n) x3]
        n_m = recs[0][0][2] // self.d                   # gate rows (m)
        d_m = recs[0][2][2] // 768                      # down rows (2048)

        def gemv(i: int, part: int, inp: torch.Tensor):
            sc, pk, n = recs[i][part]
            return kernel.q4_gemv(inp, pk, sc, n // inp.numel(), inp.numel())

        futs_g = [ex.submit(gemv, i, 0, x) for i in range(len(experts))]
        futs_u = [ex.submit(gemv, i, 1, x) for i in range(len(experts))]
        g_out = [f.result() for f in futs_g]
        u_out = [f.result() for f in futs_u]
        acts = [F.silu(g) * u for g, u in zip(g_out, u_out)]
        futs = [ex.submit(gemv, i, 2, acts[i]) for i in range(len(experts))]
        d_out = [f.result() for f in futs]
        out = (w[0].float().unsqueeze(1)
               * torch.stack(d_out)).sum(0)
        return out.unsqueeze(0).unsqueeze(0).to(h.dtype)

    def _ffn_prefill(self, l: int, h: torch.Tensor, pool) -> torch.Tensor:
        """Union-aware prefill FFN.

        Raw-Q4 pools: fused q4_gemm kernel per unique expert (no dequant
        materialisation), experts dispatched on the shared executor.
        Dequant pools: batched torch path (fallback).
        """
        T = h.shape[1]
        idx, w = self.route(l, h)                       # [T, k]
        flat = idx.reshape(-1)
        uniq, inverse = torch.unique(flat, return_inverse=True)
        k_n = idx.shape[1]
        row_of = torch.arange(T).repeat_interleave(k_n)
        flat_w = w.reshape(-1)
        out = torch.zeros(1, T, self.d, dtype=h.dtype)
        X = h[0].float()                                # [T, d] contiguous
        d_hid = X.shape[1]
        raw = isinstance(pool.ensure(l, int(uniq[0].item()))[0], tuple)

        from jouleai.native import kernel

        def expert_block(j: int, e: int):
            recs = pool.ensure(l, e)
            if raw:
                sc_g, pk_g, n_g = recs[0]
                sc_u, pk_u, n_u = recs[1]
                sc_d, pk_d, n_d = recs[2]
                m_g = n_g // d_hid
                G = torch.empty(T, m_g, dtype=torch.float32)
                kernel.q4_gemm(G, X, pk_g, sc_g, m_g, d_hid)
                U = torch.empty(T, m_g, dtype=torch.float32)
                kernel.q4_gemm(U, X, pk_u, sc_u, m_g, d_hid)
                act = F.silu(G) * U
                D = torch.empty(T, d_hid, dtype=torch.float32)
                kernel.q4_gemm(D, act, pk_d, sc_d, d_hid, m_g)
            else:
                g, u, dn = recs
                g, u, dn = g.to(h.dtype), u.to(h.dtype), dn.to(h.dtype)
                act = F.silu(X.to(h.dtype) @ g.T) * (X.to(h.dtype) @ u.T)
                D = act @ dn.T
                D = D.float()
            sel = (inverse == j).nonzero(as_tuple=True)[0]
            rows = row_of[sel]
            y = flat_w[sel].float().unsqueeze(1) * D[rows]
            return rows, y

        ex = getattr(self, "_executor", None)
        n_uniq = len(uniq)
        if ex is not None and n_uniq > 1:
            results = list(ex.map(lambda j: expert_block(j, int(uniq[j].item())),
                                  range(n_uniq)))
        else:
            results = [expert_block(j, int(uniq[j].item())) for j in range(n_uniq)]
        for rows, y in results:
            out[0].index_add_(0, rows, y.to(h.dtype))
        return out

    def ffn_dense(self, l: int, h: torch.Tensor) -> torch.Tensor:
        p = f"model.layers.{l}.mlp"
        g = self.store.full(f"{p}.gate_proj.weight")
        u = self.store.full(f"{p}.up_proj.weight")
        dn = self.store.full(f"{p}.down_proj.weight")
        return (F.silu(h @ g.T) * (h @ u.T)) @ dn.T

    def forward(self, ids, cache: dict, start_pos: int, pool=None) -> torch.Tensor:
        input_ids = ids["input_ids"] if not torch.is_tensor(ids) else ids
        x = self.embed[input_ids[0]].unsqueeze(0)
        for l in range(self.n_layers):
            h = self.rms(x, self.norm1[l], self.eps)
            x = x + self.attn(l, h, cache, start_pos)
            h = self.rms(x, self.norm2[l], self.eps)
            x = x + (self.ffn_moe(l, h, pool) if self.is_moe else self.ffn_dense(l, h))
        return self.rms(x, self.final_norm, self.eps) @ self.lm_head.T

    def generate(self, ids, max_new: int, pool, eos_id: int | None = None):
        cache: dict = {}
        t0 = time.perf_counter()
        logits = self.forward(ids, cache, 0, pool)
        if pool is not None:
            pool.prune_to_budget()          # keep highest-usage experts for decode
        p_len = (ids["input_ids"] if not torch.is_tensor(ids) else ids).shape[1]
        out = [int(logits[0, -1].argmax())]
        io0 = pool.stats.io_bytes if pool is not None else 0
        for i in range(max_new - 1):
            if eos_id is not None and out[-1] == eos_id:
                break
            step = torch.tensor([[out[-1]]])
            logits = self.forward(step, cache, p_len + i, pool)
            out.append(int(logits[0, -1].argmax()))
        wall = time.perf_counter() - t0
        n = max(len(out) - 1, 1)
        r = {"ids": out, "tok_s": len(out) / wall}
        if pool is not None:
            r.update({
                "io_mb_per_tok": (pool.stats.io_bytes - io0) / n / 1048576,
                "hit": pool.stats.hit_rate,
                "resident_mb": pool.resident_bytes() / 1048576,
            })
        return r


class NamedExpertPool:
    def __init__(self, store: SenseWeightStore, n_layers: int, n_experts: int,
                 budget_bytes: int):
        self.store = store
        self._inner = ExpertLRUPool(self, budget_bytes)
        self._sizes = [
            [sum(store.bytes_of(f"model.layers.{l}.mlp.experts.{e}.{p}_proj.weight")
                 for p in ("gate", "up", "down")) for e in range(n_experts)]
            for l in range(n_layers)
        ]

    def expert_size_bytes(self, l: int, e: int) -> int:
        return self._sizes[l][e]

    def gather(self, l: int, e: int):
        p = f"model.layers.{l}.mlp.experts.{e}"
        return tuple(self.store.full(f"{p}.{x}_proj.weight") for x in ("gate", "up", "down"))

    def __getattr__(self, name):
        return getattr(self._inner, name)


def prefix_pct(a: list[int], b: list[int]) -> float:
    same = 0
    for x, y in zip(a, b):
        if x != y:
            break
        same += 1
    return 100.0 * same / max(len(b), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense-validate", "moe-stream"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--budget-gb", type=float, nargs="+", default=[8.0, 16.0])
    ap.add_argument("--q4", action="store_true", help="serve experts from Q4 store")
    ap.add_argument("--keep-pool", action="store_true",
                    help="persist the pool across queries (production-like warm serve)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "dense-validate":
        md = Path(args.model or "models/Qwen3-8B")
        tok = __import__("transformers").AutoTokenizer.from_pretrained(md)
        eos = tok.eos_token_id
        eng = Qwen3Streamer(md)
        results = {"mode": "dense-validate", "model": str(md), "rows": []}

        def prompt_ids(q):
            t = tok.apply_chat_template([{"role": "user", "content": q}],
                                        add_generation_prompt=True, tokenize=False)
            return tok(t, return_tensors="pt")

        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(md, dtype=torch.bfloat16)
        hf.eval()
        refs = {}
        for key, q in QUERIES.items():
            ids = prompt_ids(q)
            with torch.no_grad():
                o = hf.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                pad_token_id=eos)
            refs[key] = o[0, ids["input_ids"].shape[1]:].tolist()
        with torch.no_grad():
            ids = prompt_ids(QUERIES["easy"])
            lg = eng.forward(ids, {}, 0, None)[0, -1].float()
            ids2 = prompt_ids(QUERIES["easy"])
            hf_lg = hf(**ids2).logits[0, -1].float()
        print(f"logit diff: {(lg - hf_lg).abs().max().item():.4f} "
              f"argmax match: {int(lg.argmax()) == int(hf_lg.argmax())}")
        del hf
        gc.collect()
        for key, q in QUERIES.items():
            r = eng.generate(prompt_ids(q), args.max_new, None, eos)
            ident = r["ids"][: len(refs[key])] == refs[key]
            ans = tok.decode(r["ids"], skip_special_tokens=True)
            results["rows"].append({"key": key, "identical": ident, "answer": ans,
                                    "tok_s": r["tok_s"]})
            print(f"  [{key}] identical={ident} {r['tok_s']:.2f} tok/s | {ans[:60]!r}")
        Path("results/proto_qwen3_dense.json").write_text(json.dumps(results, indent=2))
        return

    # ---- moe-stream ----
    md = Path(args.model or "models/Qwen3-30B-A3B-Instruct-2507")
    tok = __import__("transformers").AutoTokenizer.from_pretrained(md)
    eos = tok.eos_token_id
    eng = Qwen3Streamer(md)
    total_experts = eng.n_layers * eng.cfg["num_experts"]
    print(f"engine ready: {eng.n_layers} layers x {eng.cfg['num_experts']} experts "
          f"(top-{eng.cfg['num_experts_per_tok']}) = {total_experts} experts on disk",
          flush=True)
    results = {"mode": "moe-stream", "model": str(md), "q4": bool(args.q4),
               "keep_pool": bool(args.keep_pool), "rows": []}

    def prompt_ids(q):
        t = tok.apply_chat_template([{"role": "user", "content": q}],
                                    add_generation_prompt=True, tokenize=False)
        return tok(t, return_tensors="pt")

    prev: dict[str, list[int]] = {}
    for bgb in args.budget_gb:
        if args.q4:
            from jouleai.storage.q4_store import Q4ExpertPool
            pool = Q4ExpertPool(md, eng.n_layers, eng.cfg["num_experts"],
                                int(bgb * 1073741824))
        else:
            pool = NamedExpertPool(eng.store, eng.n_layers, eng.cfg["num_experts"],
                                   int(bgb * 1073741824))
        for key, q in QUERIES.items():
            if not args.keep_pool:
                pool.clear()
                pool._inner.stats = type(pool._inner.stats)()
            r = eng.generate(prompt_ids(q), args.max_new, pool, eos)
            ans = tok.decode(r["ids"], skip_special_tokens=True)
            same_vs_prev = (key in prev and r["ids"][: len(prev[key])]
                            == prev[key][: len(r["ids"])])
            row = {"budget_gb": bgb, "key": key, "answer": ans, **r,
                   "identical_vs_prev_budget": same_vs_prev}
            results["rows"].append(row)
            print(f"  [{bgb:4.0f}GB/{key:6s}] {r['tok_s']:.2f} tok/s | hit {r['hit']:.0%} | "
                  f"IO {r['io_mb_per_tok']:.0f} MB/tok | resident {r['resident_mb']:.0f} MB | "
                  f"same_as_prev_budget={same_vs_prev}", flush=True)
            if key not in prev or bgb == max(args.budget_gb):
                prev[key] = r["ids"]
        del pool
        gc.collect()

    out = args.out or "results/proto_qwen3moe_stream.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
