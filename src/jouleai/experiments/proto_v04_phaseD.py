"""Phase D experiment: trained probe predictor (the converter's mask oracle).

Steps:
  1. Calibration: diverse prompts -> engine collects (hidden, act) per layer
  2. Train: closed-form ridge probes W_l: act ≈ h @ W   (one-time converter step)
  3. Evaluate: probe-mask IoU vs exact mask on held-out prompts
  4. E2E: E/M/H queries, probe mode vs static mode (margin-verify gate on both)

Run:  python src/jouleai/experiments/proto_v04_phaseD.py --model models/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.engine.stream_engine import StreamEngine  # noqa: E402
from jouleai.routing.mask_policy import TopMassPolicy  # noqa: E402
from jouleai.routing.probe_bank import ProbeBank, train_probes  # noqa: E402

QUERIES = {
    "easy": "What is the capital of France? Answer in one sentence.",
    "medium": "Explain photosynthesis in simple terms, in about 3 sentences.",
    "hard": (
        "A train travels 120 km in 90 minutes. Its speed then halves. "
        "How long will it take to travel another 60 km? Reason step by step, "
        "then give the final answer."
    ),
}

CALIBRATION = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms.",
    "A train travels 120 km in 90 minutes, then its speed halves. How long for another 60 km?",
    "Write a Python function that reverses a string.",
    "List five healthy breakfast ideas with short reasons.",
    "Summarize what DNS does in three sentences.",
    "Translate 'good morning, how are you' into Burmese.",
    "Why does ice float on water? Explain with density.",
    "Compare REST and GraphQL APIs in a short table.",
    "Write a haiku about autumn rain.",
    "What causes the seasons on Earth?",
    "Debug this code: for i in range(10) print(i) — what is wrong?",
    "Explain recursion to a ten-year-old using a story.",
    "Give the steps to make a cup of green tea.",
    "What is the difference between TCP and UDP?",
    "Solve: 17 * 24 - 8 / 2.",
    "Describe the water cycle step by step.",
    "Write a SQL query to find the top 5 customers by total orders.",
    "Explain what a neural network layer does.",
    "Propose three names for a coffee shop and explain each.",
    "How do vaccines work, in simple words?",
    "Convert 120 kilometers per hour into meters per second.",
    "Tell a two-sentence bedtime story about a moon rabbit.",
    "Explain gradient descent with a mountain analogy.",
]
HOLDOUT = [
    "What is the largest planet in the solar system?",
    "Explain how HTTPS keeps data safe, briefly.",
    "If a shirt costs 25 dollars after a 20 percent discount, what was the original price?",
    "Write a regex that matches an email address.",
    "Why do leaves change color in autumn?",
]


def prompt_ids(tok, query: str):
    text = tok.apply_chat_template(
        [{"role": "user", "content": query}], add_generation_prompt=True, tokenize=False
    )
    return tok(text, return_tensors="pt")


def collect_xy(eng: StreamEngine, tok, prompts: list[str]):
    """Run full-FFN forwards, collect per-layer (h, act) pairs across prompts."""
    xs: list[list[torch.Tensor]] = [[] for _ in range(eng.n_layers)]
    ys: list[list[torch.Tensor]] = [[] for _ in range(eng.n_layers)]
    for p in prompts:
        eng.collect = []
        with torch.no_grad():
            eng.forward(prompt_ids(tok, p), {}, 0, None, None)
        for l, (h, act) in enumerate(eng.collect):
            xs[l].append(h)
            ys[l].append(act)
    eng.collect = None
    return [torch.cat(x) for x in xs], [torch.cat(y) for y in ys]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--verify-k", type=int, default=8)
    ap.add_argument("--mass", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--probe-every", type=int, default=4,
                    help="probe mask refresh cadence (tokens)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model)
    down_t = ROOT / "storage" / "converted" / model_dir.name / "down_t.safetensors"
    probes_path = ROOT / "storage" / "converted" / model_dir.name / "probes.safetensors"
    results: dict = {"model": str(model_dir), "mass": args.mass,
                     "max_new": args.max_new, "verify_k": args.verify_k}

    print("loading HF reference ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    hf.eval()
    eos = tok.eos_token_id
    refs: dict[str, tuple[str, list[int]]] = {}
    for key, q in QUERIES.items():
        ids = prompt_ids(tok, q)
        with torch.no_grad():
            out = hf.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                              pad_token_id=eos)
        new = out[0, ids["input_ids"].shape[1]:]
        refs[key] = (tok.decode(new, skip_special_tokens=True), new.tolist())
    del hf
    gc.collect()

    eng = StreamEngine(model_dir, down_t_dir=down_t, policy=TopMassPolicy(args.mass))

    # -------- 1-2: train probes (one-time converter step) ------------------
    if not args.skip_train and not probes_path.exists():
        print(f"collecting calibration ({len(CALIBRATION)} prompts) ...", flush=True)
        t0 = time.perf_counter()
        xs, ys = collect_xy(eng, tok, CALIBRATION)
        print(f"  collected in {time.perf_counter()-t0:.0f}s "
              f"({xs[0].shape[0]} samples/layer)", flush=True)
        weights = {}
        for l in range(eng.n_layers):
            t0 = time.perf_counter()
            W = train_probes(xs[l].float(), ys[l].float(), lam_frac=args.lam)
            weights[f"model.layers.{l}.probe.weight"] = W.bfloat16()
            print(f"  layer {l:2d}: W {tuple(W.shape)} trained in "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
        ProbeBank(weights).save(probes_path)
        print(f"  saved {probes_path}", flush=True)
        del xs, ys, weights
        gc.collect()

    probes = ProbeBank.load(probes_path)
    results["n_probes"] = len(probes)

    # -------- 3: probe mask quality on held-out prompts --------------------
    print("evaluating probe mask IoU on held-out prompts ...", flush=True)
    ious_per_layer = [[] for _ in range(eng.n_layers)]
    xs_h, ys_h = collect_xy(eng, tok, HOLDOUT)
    with torch.no_grad():
        for l in range(eng.n_layers):
            preds = (xs_h[l].float() @ probes.layers[l].float())
            for j in range(preds.shape[0]):
                m_pred = eng.policy.mask(preds[j].abs())
                m_true = eng.policy.mask(ys_h[l][j].abs())
                inter = (m_pred & m_true).sum().item()
                union = (m_pred | m_true).sum().item()
                ious_per_layer[l].append(inter / max(union, 1))
    ious = [sum(v) / len(v) for v in ious_per_layer]
    results["mask_iou_mean"] = sum(ious) / len(ious)
    results["mask_iou_first_last"] = [ious[0], ious[-1]]
    print(f"  probe mask IoU: mean {results['mask_iou_mean']:.3f} "
          f"(L0 {ious[0]:.3f} ... L{l+1} {ious[-1]:.3f})", flush=True)

    # -------- 4: E2E probe vs static ----------------------------------------
    def prefix_pct(a: list[int], b: list[int]) -> float:
        same = 0
        for x, y in zip(a, b):
            if x != y:
                break
            same += 1
        return 100.0 * same / max(len(b), 1)

    rows = []
    for key, q in QUERIES.items():
        ids = prompt_ids(tok, q)
        row: dict = {"key": key, "hf_ref": refs[key][0]}
        for label, kwargs in [
            ("static", dict(mask_source="static")),
            ("probe", dict(mask_source="probe", probes=probes,
                           refresh_every=args.probe_every)),
        ]:
            t0 = time.perf_counter()
            g = eng.generate(ids, args.max_new, keep=0.9, verify_k=args.verify_k,
                             eos_id=eos, **kwargs)
            wall = time.perf_counter() - t0
            row[label] = {
                "answer": tok.decode(g["ids"], skip_special_tokens=True),
                "prefix_identical_pct": prefix_pct(g["ids"], refs[key][1]),
                "gate": g["gate"],
                "tok_s_eff": 1.0 / max(g["decode_s_per_tok"], 1e-9),
                "wall_s": wall,
                "pool_touched_mb": g["pool_touched_mb"],
                "keep_mean": g["keep_mean"],
                "pool_gathered_mb": g["pool_gathered_mb"],
            }
            r = row[label]
            print(f"  [{key}/{label}] prefix {r['prefix_identical_pct']:5.1f}% | "
                  f"{r['tok_s_eff']:.2f} tok/s | pool {r['pool_touched_mb']:.0f} MB | "
                  f"gate: {r['gate'].get('note','off')}", flush=True)
        rows.append(row)
    results["rows"] = rows

    print("\n================ PHASE D SUMMARY ================")
    for row in rows:
        s, p = row["static"], row["probe"]
        fb = lambda r: "FB" if r["gate"].get("fell_back") else "ok"  # noqa: E731
        print(f"{row['key']:7s} static: {fb(s)} {s['tok_s_eff']:5.2f}t/s | "
              f"probe: {fb(p)} {p['tok_s_eff']:5.2f}t/s | IoU {results['mask_iou_mean']:.2f}")

    out = args.out or f"results/proto_phaseD_{model_dir.name}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
