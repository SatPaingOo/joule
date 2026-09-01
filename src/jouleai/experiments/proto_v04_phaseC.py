"""Phase C experiment: static mask vs adaptive mask refresh (the predictor part).

For each query (easy/medium/hard), two runs on the same StreamEngine:
  static    refresh_every=0   (Entry 17 baseline: mask fixed from prefill)
  adaptive  refresh_every=N   (every N-th decode token is an exact full-FFN step
                              whose activations refresh masks + delta-extend pool)

Measured: verify-gate fallback rate, token-identity vs HF, effective tok/s
(decode incl. refresh cost), pool bytes + refresh scan bytes.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--verify-k", type=int, default=8)
    ap.add_argument("--mass", type=float, default=0.9)
    ap.add_argument("--refresh", type=int, nargs="+", default=[6])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model)
    down_t = ROOT / "storage" / "converted" / model_dir.name / "down_t.safetensors"
    results: dict = {"model": str(model_dir), "max_new": args.max_new,
                     "mass": args.mass, "verify_k": args.verify_k, "rows": []}

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

    eng = StreamEngine(model_dir, down_t_dir=down_t,
                       policy=TopMassPolicy(args.mass))

    def prefix_pct(a: list[int], b: list[int]) -> float:
        same = 0
        for x, y in zip(a, b):
            if x != y:
                break
            same += 1
        return 100.0 * same / max(len(b), 1)

    for key, q in QUERIES.items():
        ids = prompt_ids(tok, q)
        row: dict = {"key": key, "hf_ref": refs[key][0]}

        for label, refresh in [("static", 0)] + [(f"adaptive{r}", r) for r in args.refresh]:
            t0 = time.perf_counter()
            g = eng.generate(ids, args.max_new, keep=0.9, verify_k=args.verify_k,
                             eos_id=eos, refresh_every=refresh)
            wall = time.perf_counter() - t0
            pct = prefix_pct(g["ids"], refs[key][1])
            row[label] = {
                "answer": tok.decode(g["ids"], skip_special_tokens=True),
                "prefix_identical_pct": pct,
                "token_identical": g["ids"][: len(refs[key][1])] == refs[key][1],
                "gate": g["gate"],
                "tok_s_eff": 1.0 / max(g["decode_s_per_tok"], 1e-9),
                "wall_s": wall,
                "pool_touched_mb": g["pool_touched_mb"],
                "refresh_scan_mb": g["refresh"]["refresh_scan_mb"],
                "n_refresh": g["refresh"]["n_refresh"],
                "added_rows": g["refresh"]["added_rows"],
                "keep_mean_final": g["keep_mean"],
            }
            r = row[label]
            print(f"  [{key}/{label}] prefix {pct:5.1f}% | {r['tok_s_eff']:.2f} tok/s | "
                  f"gate: {r['gate'].get('note', 'off')}", flush=True)
        results["rows"].append(row)

    print("\n================ PHASE C SUMMARY ================")
    labels = ["static"] + [f"adaptive{r}" for r in args.refresh]
    print(f"{'query':8s}" + "".join(f" | {lb:>22s}" for lb in labels))
    for row in results["rows"]:
        line = f"{row['key']:8s}"
        for lb in labels:
            r = row[lb]
            fell = r["gate"].get("fell_back", False)
            tag = "FALLBACK" if fell else "verified"
            line += f" | {r['prefix_identical_pct']:5.1f}% {tag:>9s} {r['tok_s_eff']:5.2f}t/s"
        print(line)
    fb = {lb: sum(1 for row in results["rows"] if row[lb]["gate"].get("fell_back"))
          for lb in labels}
    print(f"fallback rate: " + "  ".join(f"{lb}={fb[lb]}/3" for lb in labels))

    out = args.out or f"results/proto_phaseC_{model_dir.name}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
