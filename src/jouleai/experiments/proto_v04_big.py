"""Prototype v0.4 experiment: database-style masked serving on a big model.

Paths measured per query (easy / medium / hard):
  A  full baseline generation (also captures per-layer masks + sparsity)
  C  masked generation reusing the query's own captured masks (oracle JIT)
  D  masked generation on a SIMILAR query reusing the canonical masks (mask reuse)
  E  exact repeat served from the answer cache (no model compute)

Resources (RAM / CPU / battery-W) are sampled per generation by ResourceMonitor.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.engine.masked_mlp import MaskedMLPController  # noqa: E402
from jouleai.monitor.resource_monitor import ResourceMonitor  # noqa: E402
from jouleai.routing.mask_policy import TopMassPolicy  # noqa: E402
from jouleai.store.mask_cache import ActivationMaskCache, embed_query  # noqa: E402

QUERIES = {
    "easy": "What is the capital of France? Answer in one sentence.",
    "medium": "Explain photosynthesis in simple terms, in about 3 sentences.",
    "hard": (
        "A train travels 120 km in 90 minutes. Its speed then halves. "
        "How long will it take to travel another 60 km? Reason step by step, "
        "then give the final answer."
    ),
}
SIMILAR = {
    "easy": "Which city is the capital of France? Answer in one sentence.",
    "medium": "Describe how photosynthesis works in simple words, about 3 sentences.",
    "hard": (
        "A train covers 120 km in 1.5 hours. Then its speed is cut in half. "
        "How much time does it need for the next 60 km? Reason step by step "
        "and give the final answer."
    ),
}


def prompt_ids(tok, query: str):
    text = tok.apply_chat_template(
        [{"role": "user", "content": query}], add_generation_prompt=True, tokenize=False
    )
    return tok(text, return_tensors="pt")


def generate(model, tok, query: str, max_new: int) -> tuple[str, dict]:
    """Greedy generation with per-phase timing."""
    ids = prompt_ids(tok, query)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = time.perf_counter() - t0
    new_tokens = out.shape[1] - ids["input_ids"].shape[1]
    text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, {
        "prefill_prompt_tokens": int(ids["input_ids"].shape[1]),
        "new_tokens": int(new_tokens),
        "wall_s": dt,
        "tok_per_s": new_tokens / dt,
    }


def token_drift(ref: str, test: str, tok) -> dict:
    a = tok(ref, add_special_tokens=False)["input_ids"]
    b = tok(test, add_special_tokens=False)["input_ids"]
    same = 0
    for x, y in zip(a, b):
        if x != y:
            break
        same += 1
    return {
        "identical_prefix_pct": 100.0 * same / max(len(a), 1),
        "text_ratio": difflib.SequenceMatcher(None, ref, test).ratio(),
        "token_identical": a == b,
    }


def run_generation_path(model, tok, ctrl, tasks, max_new, label) -> list[dict]:
    """Run a list of (key, query, masks|None) through generate with monitoring."""
    rows = []
    for key, query, masks in tasks:
        if masks is not None:
            ctrl.start_apply(masks)
        else:
            ctrl.stop()
        with ResourceMonitor(interval_s=0.5, battery_every_n=10) as mon:
            text, timing = generate(model, tok, query, max_new)
        rep = mon.report()
        keep = [float((m.sum() / m.numel()).item()) if m is not None else 1.0
                for m in (masks or [None] * len(ctrl.layers))]
        rows.append({
            "key": key, "query": query, "answer": text, **timing,
            "ffn_keep_mean": sum(keep) / len(keep),
            "res": rep.__dict__,
            "res_summary": rep.summary(),
        })
        print(f"  [{label}] {key}: {timing['tok_per_s']:.2f} tok/s, "
              f"keep {rows[-1]['ffn_keep_mean']:.0%}, {rep.summary()}", flush=True)
    ctrl.stop()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--mass", type=float, default=0.9, help="TopMass fraction")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"loading {args.model} ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    print(f"loaded in {time.perf_counter() - t0:.0f}s", flush=True)

    ctrl = MaskedMLPController(model)
    policy = TopMassPolicy(args.mass)
    max_new = args.tokens

    results = {"model": args.model, "mass": args.mass, "tokens": max_new, "paths": {}}

    # ---------------- Path A: full baseline + mask capture ----------------
    print("\n=== Path A: FULL baseline + mask capture ===", flush=True)
    rows_a, mask_bank = [], {}
    with ResourceMonitor(interval_s=0.5, battery_every_n=10) as mon_all:
        for key, query in QUERIES.items():
            ctrl.start_capture()
            text, timing = generate(model, tok, query, max_new)
            captured = ctrl.captured
            ctrl.stop()

            masks = [policy.mask(a) for a in captured]
            keep = [float((m.sum() / m.numel()).item()) for m in masks]
            mask_bank[key] = masks
            rows_a.append({"key": key, "query": query, "answer": text, **timing,
                           "ffn_keep_mean": sum(keep) / len(keep)})
            print(f"  [A] {key}: {timing['tok_per_s']:.2f} tok/s, "
                  f"sparsity keep {sum(keep) / len(keep):.1%}", flush=True)
            del captured
            gc.collect()
    rep_all = mon_all.report()
    results["paths"]["A_full"] = {"rows": rows_a, "res": rep_all.__dict__}

    # ---------------- Path C: masked with own masks (oracle JIT) -----------
    print("\n=== Path C: MASKED, own masks (oracle JIT) ===", flush=True)
    tasks_c = [(k, QUERIES[k], mask_bank[k]) for k in QUERIES]
    results["paths"]["C_masked_own"] = {"rows": run_generation_path(
        model, tok, ctrl, tasks_c, max_new, "C")}

    # ---------------- Path D: full baseline for similar queries ------------
    print("\n=== Path D-baseline: FULL on similar queries ===", flush=True)
    tasks_d0 = [(k + "2", SIMILAR[k], None) for k in QUERIES]
    results["paths"]["D0_full_similar"] = {"rows": run_generation_path(
        model, tok, ctrl, tasks_d0, max_new, "D0")}

    # ---------------- Path D: masked reuse on similar queries --------------
    print("\n=== Path D: MASKED, reuse canonical masks ===", flush=True)
    tasks_d = [(k + "2", SIMILAR[k], mask_bank[k]) for k in QUERIES]
    rows_d = run_generation_path(model, tok, ctrl, tasks_d, max_new, "D")
    for r in rows_d:
        base = next(x for x in results["paths"]["D0_full_similar"]["rows"]
                    if x["key"] == r["key"])
        r["drift"] = token_drift(base["answer"], r["answer"], tok)
        r["speedup_vs_full_similar"] = base["wall_s"] / r["wall_s"]
    results["paths"]["D_masked_reuse"] = {"rows": rows_d}

    # drift of C vs A
    for r in results["paths"]["C_masked_own"]["rows"]:
        base = next(x for x in rows_a if x["key"] == r["key"])
        r["drift"] = token_drift(base["answer"], r["answer"], tok)
        r["speedup_vs_full"] = base["wall_s"] / r["wall_s"]

    # ---------------- Path E: answer cache exact repeat --------------------
    print("\n=== Path E: ANSWER CACHE exact repeat ===", flush=True)
    norm = lambda s: s.strip().lower()  # noqa: E731
    cache: dict[str, str] = {norm(q): next(x["answer"] for x in rows_a if x["key"] == k)
                             for k, q in QUERIES.items()}
    q = QUERIES["easy"]
    hit = cache.get(norm(q))
    lookup_ms = 0.0
    for _ in range(1000):  # stable latency estimate
        t0 = time.perf_counter()
        cache.get(norm(q))
        lookup_ms += time.perf_counter() - t0
    lookup_ms = lookup_ms * 1000 / 1000
    results["paths"]["E_cache"] = {
        "query": q, "hit": hit is not None, "lookup_ms": lookup_ms,
        "answer_matches_baseline": hit == rows_a[0]["answer"],
    }
    print(f"  [E] hit={hit is not None}, lookup {lookup_ms:.4f} ms", flush=True)

    # ---------------- mask cache signature demo ---------------------------
    sig = embed_query(tok, model, SIMILAR["easy"])
    mc = ActivationMaskCache(similarity_threshold=0.85)
    mc.put(embed_query(tok, model, QUERIES["easy"]), mask_bank["easy"], QUERIES["easy"])
    mc.put(embed_query(tok, model, QUERIES["hard"]), mask_bank["hard"], QUERIES["hard"])
    got = mc.get(sig)
    results["mask_cache_similarity"] = {
        "similar_query_matches_easy_mask": got is not None and "easy" in mc.queries[got[0]],
        "similarity": got[1] if got else None,
    }
    print(f"  mask-cache: similar query -> {got}", flush=True)

    # ---------------- summary ---------------------------------------------
    print("\n================ SUMMARY ================")
    for k in QUERIES:
        a = next(x for x in rows_a if x["key"] == k)
        c = next(x for x in results["paths"]["C_masked_own"]["rows"] if x["key"] == k)
        d0 = next(x for x in results["paths"]["D0_full_similar"]["rows"] if x["key"] == k + "2")
        d = next(x for x in rows_d if x["key"] == k + "2")
        print(f"{k:7s} keep {a['ffn_keep_mean']:5.1%} | "
              f"C drift {c['drift']['identical_prefix_pct']:5.1f}% "
              f"({c['speedup_vs_full']:.2f}x) | "
              f"D drift {d['drift']['identical_prefix_pct']:5.1f}% "
              f"({d['speedup_vs_full_similar']:.2f}x)")
    print(f"cache lookup: {lookup_ms:.4f} ms")

    out = args.out or f"results/proto_v04_big_{Path(args.model).name}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
