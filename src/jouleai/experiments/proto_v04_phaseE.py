"""Phase E experiment: MoE-style expert streaming (machinery validation for
frontier-model serving on laptops).

Simulates MoE on a dense Qwen2.5 model: FFN neurons are partitioned into E
experts per layer; a synthetic top-k router picks experts; an ExpertLRUPool
keeps a RAM budget of experts resident (LRU eviction, disk gather on miss).
This exercises the EXACT machinery a real MoE (Kimi/GLM/DeepSeek-class) needs —
only the router changes (trained gate instead of synthetic).

Measured per query: tok/s, LRU hit rate, IO bytes/token, pool peak, drift.
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

from jouleai.engine.stream_engine import rms_norm  # noqa: E402
from jouleai.engine.stream_engine import StreamEngine  # noqa: E402
from jouleai.storage.expert_store import ExpertLRUPool, ExpertStore, MoERouter  # noqa: E402
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


def prompt_ids(tok, query: str):
    text = tok.apply_chat_template(
        [{"role": "user", "content": query}], add_generation_prompt=True, tokenize=False
    )
    return tok(text, return_tensors="pt")


class MoEForward:
    """Dense-engine weights + router-guided expert FFN (the MoE adapter v0)."""

    def __init__(self, eng: StreamEngine, store: ExpertStore, router: MoERouter,
                 pool: ExpertLRUPool):
        self.eng, self.es, self.router, self.pool = eng, store, router, pool

    def _ffn_expert(self, l: int, h: torch.Tensor, experts: torch.Tensor):
        gs, us, dns = [], [], []
        for e in experts.tolist():
            g, u, dn = self.pool.ensure(l, e)
            gs.append(g); us.append(u); dns.append(dn)
        g = torch.cat(gs); u = torch.cat(us); dn = torch.cat(dns)
        act = F.silu(h @ g.T) * (h @ u.T)          # [1, T, k*m]
        return act @ dn                             # [1, T, d]

    def forward(self, ids, cache: dict, start_pos: int) -> torch.Tensor:
        eng = self.eng
        input_ids = ids["input_ids"] if not torch.is_tensor(ids) else ids
        x = eng.embed[input_ids[0]].unsqueeze(0)
        for l in range(eng.n_layers):
            h = rms_norm(x, eng.norm1[l], eng.eps)
            x = x + eng._attn(l, h, cache, start_pos)
            h = rms_norm(x, eng.norm2[l], eng.eps)
            experts = self.router.select(l, h)      # top-k on last token (v0)
            x = x + self._ffn_expert(l, h, experts)
        return rms_norm(x, eng.final_norm, eng.eps) @ eng.lm_head.T

    def generate(self, ids, max_new: int, eos_id: int | None = None):
        cache: dict = {}
        t0 = time.perf_counter()
        logits = self.forward(ids, cache, 0)
        prompt_len = (ids["input_ids"] if not torch.is_tensor(ids) else ids).shape[1]
        out = [int(logits[0, -1].argmax())]
        io_start = self.pool.stats.io_bytes
        for i in range(max_new - 1):
            if eos_id is not None and out[-1] == eos_id:
                break
            step = torch.tensor([[out[-1]]])
            logits = self.forward(step, cache, prompt_len + i)
            out.append(int(logits[0, -1].argmax()))
        wall = time.perf_counter() - t0
        n_gen = max(len(out) - 1, 1)
        io_bytes = self.pool.stats.io_bytes - io_start
        return {
            "ids": out, "wall_s": wall, "tok_s": len(out) / wall,
            "io_mb_per_tok": io_bytes / n_gen / 1048576,
            "pool_hit": self.pool.stats.hit_rate,
            "pool_resident_mb": self.pool.resident_bytes() / 1048576,
            "pool_peak_mb": self.pool.stats.peak_resident / 1048576,
            "evictions": self.pool.stats.evictions,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--topk", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--budget-mb", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model)
    down_t = ROOT / "storage" / "converted" / model_dir.name / "down_t.safetensors"

    print("building engine (fixed weights) ...", flush=True)
    tok_ref = AutoTokenizerLite(model_dir)
    eng = StreamEngine(model_dir, down_t_dir=down_t)

    store = SenseWeightStore(model_dir)
    dt = SenseWeightStore(down_t)
    es = ExpertStore(store, dt, eng.n_layers, eng.cfg["intermediate_size"],
                     args.experts)
    results = {"model": str(model_dir), "experts": args.experts,
               "results": []}

    for k in args.topk:
        for budget_mb in args.budget_mb:
            router = MoERouter(eng.n_layers, eng.d, args.experts, k)
            pool = ExpertLRUPool(es, budget_mb * 1048576)
            moe = MoEForward(eng, es, router, pool)
            row = {"topk": k, "budget_mb": budget_mb, "queries": {}}
            for key, q in QUERIES.items():
                ids = prompt_ids(tok_ref, q)
                pool.clear()
                pool.stats = type(pool.stats)()
                r = moe.generate(ids, args.max_new)
                row["queries"][key] = r
                print(f"  [k={k:2d} pool={budget_mb:5d}MB {key:6s}] "
                      f"{r['tok_s']:.2f} tok/s | hit {r['pool_hit']:.0%} | "
                      f"IO {r['io_mb_per_tok']:.0f} MB/tok | "
                      f"resident {r['pool_resident_mb']:.0f} MB", flush=True)
            results["results"].append(row)
            del router, pool, moe
            gc.collect()

    Path(args.out or f"results/proto_phaseE_{model_dir.name}.json").write_text(
        json.dumps(results, indent=2))
    print("done", flush=True)


class AutoTokenizerLite:
    """Minimal wrapper so the experiment does not need transformers."""

    def __init__(self, model_dir: Path):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.eos_token_id = self.tok.eos_token_id

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return self.tok.apply_chat_template(
            msgs, add_generation_prompt=add_generation_prompt, tokenize=tokenize)

    def __call__(self, text, return_tensors=None):
        return self.tok(text, return_tensors=return_tensors)


if __name__ == "__main__":
    main()
