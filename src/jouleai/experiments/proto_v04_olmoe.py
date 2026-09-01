"""Real MoE validation: OLMoE-1B-7B with trained router + expert streaming.

Key property tested: for a REAL MoE, router-guided expert streaming is
mathematically EXACT — the model itself defines sparse compute (top-8 of 64
per token); we only change WHERE expert weights live (disk + LRU pool).
So sparse serve should be token-identical to full-serve (lossless by
construction, no verify gate needed), while RAM holds only the hot set.

Measured: sparse vs full-serve answers, tok/s, LRU hit, IO/token, RAM pool.
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


class NamedExpertPool:
    """ExpertLRUPool adapter for models whose experts are separate tensors."""

    def __init__(self, store: SenseWeightStore, n_layers: int, n_experts: int,
                 budget_bytes: int):
        self.store = store
        self.n_layers, self.n_experts = n_layers, n_experts
        self._inner = ExpertLRUPool(self, budget_bytes)
        self._sizes = [
            [sum(store.bytes_of(f"model.layers.{l}.mlp.experts.{e}.{p}_proj.weight")
                 for p in ("gate", "up", "down"))
             for e in range(n_experts)]
            for l in range(n_layers)
        ]

    def expert_size_bytes(self, l: int, e: int) -> int:
        return self._sizes[l][e]

    def gather(self, l: int, e: int):
        p = f"model.layers.{l}.mlp.experts.{e}"
        return (self.store.full(f"{p}.gate_proj.weight"),
                self.store.full(f"{p}.up_proj.weight"),
                self.store.full(f"{p}.down_proj.weight"))

    # delegate pool API
    def __getattr__(self, name):
        return getattr(self._inner, name)


class OlmoeStreamer:
    """Manual OLMoE forward with expert streaming (MHA, no biases, RoPE 10k)."""

    def __init__(self, model_dir: Path):
        self.cfg = json.loads((model_dir / "config.json").read_text())
        self.store = SenseWeightStore(model_dir)
        c = self.cfg
        self.n_layers = c["num_hidden_layers"]
        self.n_experts = c["num_experts"]
        self.top_k = c["num_experts_per_tok"]
        self.d = c["hidden_size"]
        self.n_heads = c["num_attention_heads"]
        self.hd = self.d // self.n_heads
        self.eps = c.get("rms_norm_eps") or 1e-5
        self.theta = c.get("rope_theta", 1e4)
        self.embed = self.store.full("model.embed_tokens.weight")
        self.lm_head = self.store.full("lm_head.weight")
        self.final_norm = self.store.full("model.norm.weight")
        self.wq, self.wk, self.wv, self.wo = [], [], [], []
        self.norm1, self.norm2, self.gate = [], [], []
        self.q_norm, self.k_norm = [], []
        for l in range(self.n_layers):
            p = f"model.layers.{l}"
            self.wq.append(self.store.full(f"{p}.self_attn.q_proj.weight"))
            self.wk.append(self.store.full(f"{p}.self_attn.k_proj.weight"))
            self.wv.append(self.store.full(f"{p}.self_attn.v_proj.weight"))
            self.wo.append(self.store.full(f"{p}.self_attn.o_proj.weight"))
            self.q_norm.append(self.store.full(f"{p}.self_attn.q_norm.weight"))
            self.k_norm.append(self.store.full(f"{p}.self_attn.k_norm.weight"))
            self.norm1.append(self.store.full(f"{p}.input_layernorm.weight"))
            self.norm2.append(self.store.full(f"{p}.post_attention_layernorm.weight"))
            self.gate.append(self.store.full(f"{p}.mlp.gate.weight"))
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.inv_freq = inv

    def rope_cs(self, pos: torch.Tensor):
        fr = pos.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((fr, fr), dim=-1)
        return emb.cos().to(torch.bfloat16)[None, None], emb.sin().to(torch.bfloat16)[None, None]

    @staticmethod
    def _rot(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    def attn(self, l: int, x: torch.Tensor, cache: dict, start_pos: int):
        eng = self
        T = x.shape[1]
        q = x @ eng.wq[l].T                            # [1, T, 2048]
        k = x @ eng.wk[l].T
        v = x @ eng.wv[l].T
        # OLMoE quirk: whole-vector QK-norm (hidden_size RMSNorm) before rope
        qn = q.float() * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eng.eps)
        kn = k.float() * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + eng.eps)
        q = (qn.to(q.dtype) * eng.q_norm[l]).view(1, T, eng.n_heads, eng.hd).transpose(1, 2)
        k = (kn.to(k.dtype) * eng.k_norm[l]).view(1, T, eng.n_heads, eng.hd).transpose(1, 2)
        v = v.view(1, T, eng.n_heads, eng.hd).transpose(1, 2)
        pos = torch.arange(start_pos, start_pos + T)
        cos, sin = eng.rope_cs(pos)                    # [1, 1, T, hd]
        q = q * cos + eng._rot(q) * sin
        k = k * cos + eng._rot(k) * sin
        if l in cache:
            cache[l] = (torch.cat([cache[l][0], k], 2), torch.cat([cache[l][1], v], 2))
        else:
            cache[l] = (k, v)
        K, V = cache[l]
        o = F.scaled_dot_product_attention(q, K, V, is_causal=(T > 1))
        return o.transpose(1, 2).reshape(1, T, self.d) @ self.wo[l].T

    def route(self, l: int, h: torch.Tensor):
        """Return per-position (indices, weights): decode -> single row."""
        logits = h @ self.gate[l].to(h.dtype).T          # [1, T, E]
        probs = torch.softmax(logits.float(), dim=-1)[0]  # [T, E]
        top = torch.topk(probs, self.top_k, dim=-1)
        return top.indices, top.values                    # [T, k]

    def ffn_moe(self, l: int, h: torch.Tensor, pool) -> torch.Tensor:
        """Per-position expert routing; weighted sum over top-k experts."""
        T = h.shape[1]
        idx, w = self.route(l, h)                        # [T, k]
        out = torch.zeros(1, T, self.d, dtype=h.dtype)
        expert_cache: dict[int, tuple] = {}
        for t in range(T):
            for j in range(self.top_k):
                e = int(idx[t, j])
                if e not in expert_cache:
                    expert_cache[e] = pool.ensure(l, e)
            x_t = h[0, t:t + 1, :]                       # [1, d]
            y = torch.zeros(1, 1, self.d, dtype=h.dtype)
            for j in range(self.top_k):
                e = int(idx[t, j])
                g, u, dn = expert_cache[e]
                g, u, dn = g.to(h.dtype), u.to(h.dtype), dn.to(h.dtype)
                act = F.silu(x_t @ g.T) * (x_t @ u.T)    # [1, 1, m]
                y += w[t, j].to(act.dtype) * (act @ dn.T)
            out[0, t] = y[0, 0]
        return out

    def forward(self, ids, cache: dict, start_pos: int, pool) -> torch.Tensor:
        input_ids = ids["input_ids"] if not torch.is_tensor(ids) else ids
        x = self.embed[input_ids[0]].unsqueeze(0)
        for l in range(self.n_layers):
            h = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            h = (h.to(x.dtype) * self.norm1[l])
            x = x + self.attn(l, h, cache, start_pos)
            h = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            h = (h.to(x.dtype) * self.norm2[l])
            x = x + self.ffn_moe(l, h, pool)
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                + self.eps)).to(x.dtype) * self.final_norm @ self.lm_head.T

    def generate(self, ids, max_new: int, pool, eos_id: int | None = None):
        cache: dict = {}
        t0 = time.perf_counter()
        logits = self.forward(ids, cache, 0, pool)
        p_len = (ids["input_ids"] if not torch.is_tensor(ids) else ids).shape[1]
        out = [int(logits[0, -1].argmax())]
        io0 = pool.stats.io_bytes
        for i in range(max_new - 1):
            if eos_id is not None and out[-1] == eos_id:
                break
            step = torch.tensor([[out[-1]]])
            logits = self.forward(step, cache, p_len + i, pool)
            out.append(int(logits[0, -1].argmax()))
        wall = time.perf_counter() - t0
        n = max(len(out) - 1, 1)
        return {
            "ids": out, "tok_s": len(out) / wall,
            "io_mb_per_tok": (pool.stats.io_bytes - io0) / n / 1048576,
            "hit": pool.stats.hit_rate,
            "resident_mb": pool.resident_bytes() / 1048576,
            "peak_mb": pool.stats.peak_resident / 1048576,
        }


def prefix_pct(a: list[int], b: list[int]) -> float:
    same = 0
    for x, y in zip(a, b):
        if x != y:
            break
        same += 1
    return 100.0 * same / max(len(b), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/OLMoE-1B-7B-0824-Instruct")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--budget-mb", type=int, default=4096)
    ap.add_argument("--skip-hf-check", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model)
    eng = OlmoeStreamer(model_dir)
    print(f"engine ready: {eng.n_layers} layers x {eng.n_experts} experts "
          f"(top-{eng.top_k})", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    eos = tok.eos_token_id

    def prompt_ids(q: str):
        text = tok.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True,
            tokenize=False)
        return tok(text, return_tensors="pt")

    results = {"model": str(model_dir), "budget_mb": args.budget_mb, "rows": []}

    # HF reference (correctness of our manual forward)
    if not args.skip_hf_check:
        print("HF reference pass ...", flush=True)
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
        hf.eval()
        ref = {}
        for key, q in QUERIES.items():
            ids = prompt_ids(q)
            with torch.no_grad():
                o = hf.generate(**ids, max_new_tokens=args.max_new,
                                do_sample=False, pad_token_id=eos)
            ref[key] = o[0, ids["input_ids"].shape[1]:].tolist()
            print(f"  [HF] {key}: {tok.decode(ref[key], skip_special_tokens=True)[:60]!r}",
                  flush=True)
        del hf
        gc.collect()
        results["hf_ref"] = {k: tok.decode(v, skip_special_tokens=True)
                             for k, v in ref.items()}

    for label, budget in [("full", 1 << 60), ("sparse", args.budget_mb * 1048576)]:
        print(f"=== {label} serve (budget {budget >> 20} MB) ===", flush=True)
        pool = NamedExpertPool(eng.store, eng.n_layers, eng.n_experts, budget)
        for key, q in QUERIES.items():
            pool.clear()
            pool._inner.stats = type(pool._inner.stats)()
            r = eng.generate(prompt_ids(q), args.max_new, pool, eos)
            ans = tok.decode(r["ids"], skip_special_tokens=True)
            row = {"label": label, "key": key, "answer": ans,
                   "hf_ref_ids": None, **r}
            if not args.skip_hf_check:
                ids_hf = ref[key]
                row["identical_prefix_pct"] = prefix_pct(r["ids"], ids_hf)
                row["token_identical"] = r["ids"][: len(ids_hf)] == ids_hf
            results["rows"].append(row)
            print(f"  [{label}/{key}] {r['tok_s']:.2f} tok/s | hit {r['hit']:.0%} | "
                  f"IO {r['io_mb_per_tok']:.0f} MB/tok | resident {r['resident_mb']:.0f} MB"
                  + (f" | vs HF prefix {row['identical_prefix_pct']:.0f}%"
                     if "identical_prefix_pct" in row else ""), flush=True)
        del pool
        gc.collect()

    out = args.out or "results/proto_olmoe_streaming.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
