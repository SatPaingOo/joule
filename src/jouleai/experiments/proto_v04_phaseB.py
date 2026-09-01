"""Phase B experiment: true row-streaming engine on the SenseWeightStore.

Measures, for easy/medium/hard queries:
  0. engine correctness: StreamEngine full-keep logits/answers vs HF reference
  1. masked streaming (keep = TopMass coverage): answer drift, keep %, decode tok/s
  2. RAM reality: fixed RSS (resident weights) vs decode-phase RSS (fixed + pool)
     vs released RSS (post trim) — the "load what you need, release after" proof
  3. verify gate (first-k tokens recomputed with full FFN; fallback on divergence)

Run:  python src/jouleai/experiments/proto_v04_phaseB.py --model models/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.engine.stream_engine import StreamEngine  # noqa: E402
from jouleai.monitor.resource_monitor import ResourceMonitor  # noqa: E402

QUERIES = {
    "easy": "What is the capital of France? Answer in one sentence.",
    "medium": "Explain photosynthesis in simple terms, in about 3 sentences.",
    "hard": (
        "A train travels 120 km in 90 minutes. Its speed then halves. "
        "How long will it take to travel another 60 km? Reason step by step, "
        "then give the final answer."
    ),
}
PROC = psutil.Process()


def trim_ws() -> None:
    """Ask Windows to drop this process's working set (pages go to standby)."""
    try:
        psapi = ctypes.WinDLL("psapi")
        k32 = ctypes.WinDLL("kernel32")
        psapi.EmptyWorkingSet(k32.GetCurrentProcess())
    except Exception:
        pass


def prompt_ids(tok, query: str):
    text = tok.apply_chat_template(
        [{"role": "user", "content": query}], add_generation_prompt=True, tokenize=False
    )
    return tok(text, return_tensors="pt")


def build_down_t(model_dir: Path) -> Path:
    """One-time converter step: transposed down_proj for neuron-granular access."""
    out_dir = ROOT / "storage" / "converted" / model_dir.name
    out = out_dir / "down_t.safetensors"
    if out.exists():
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    from jouleai.storage.weight_store import SenseWeightStore

    store = SenseWeightStore(model_dir)
    tensors = {}
    print("converting down_proj -> column-major (down_t) ...", flush=True)
    for l in range(json.loads((model_dir / "config.json").read_text())["num_hidden_layers"]):
        name = f"model.layers.{l}.mlp.down_proj.weight"
        dn = store.full(name)
        tensors[name.replace("down_proj", "down_proj_t")] = dn.T.contiguous()
    save_file(tensors, str(out))
    print(f"  saved {out} ({out.stat().st_size / 1e9:.1f} GB)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--mass", type=float, default=0.9)
    ap.add_argument("--verify-k", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model)
    results: dict = {"model": str(model_dir), "max_new": args.max_new,
                     "mass": args.mass, "verify_k": args.verify_k, "rows": []}

    # ---------------- HF reference ----------------------------------------
    print("loading HF reference ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    hf.eval()
    eos = tok.eos_token_id

    refs: dict[str, tuple[str, list[int]]] = {}
    hf_logits_probe: dict[str, torch.Tensor] = {}
    for key, q in QUERIES.items():
        ids = prompt_ids(tok, q)
        with torch.no_grad():
            out = hf.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                              pad_token_id=eos)
        new = out[0, ids["input_ids"].shape[1]:]
        refs[key] = (tok.decode(new, skip_special_tokens=True),
                     new.tolist())
        with torch.no_grad():
            hf_logits_probe[key] = hf(**ids).logits[0, -1].float().clone()
        print(f"  [HF ref] {key}: {refs[key][0][:60]!r}", flush=True)

    # ---------------- converted layout ------------------------------------
    down_t = build_down_t(model_dir)

    # ---------------- engine ----------------------------------------------
    print("building StreamEngine (fixed weights resident) ...", flush=True)
    eng = StreamEngine(model_dir, down_t_dir=down_t, policy=__import__(
        "jouleai.routing.mask_policy", fromlist=["TopMassPolicy"]).TopMassPolicy(args.mass))
    print(f"  engine built", flush=True)

    # correctness probe: engine vs HF logits at last prompt position (full compute)
    probe = {}
    for key, q in QUERIES.items():
        ids = prompt_ids(tok, q)
        with torch.no_grad():
            lg = eng.forward(ids, {}, 0, None, None)
        diff = (lg[0, -1].float() - hf_logits_probe[key]).abs().max().item()
        top_eng = int(lg[0, -1].argmax())
        top_hf = int(hf_logits_probe[key].argmax())
        probe[key] = {"max_abs_logit_diff": diff, "argmax_match": top_eng == top_hf}
        print(f"  [probe] {key}: max|dlogit|={diff:.4f} argmax_match={top_eng == top_hf}",
              flush=True)
    results["correctness_probe"] = probe
    del hf_logits_probe

    # free HF before RSS measurements
    del hf
    gc.collect()
    trim_ws()
    time.sleep(1)
    fixed_mb = PROC.memory_info().rss / 1048576

    # ---------------- per query: full-engine + masked + verify ------------
    for key, q in QUERIES.items():
        ids = prompt_ids(tok, q)
        row: dict = {"key": key}

        # (a) engine full-keep answer -> correctness vs HF ref
        t0 = time.perf_counter()
        full = eng.generate(ids, args.max_new, keep=1.0, eos_id=eos)
        full_ids = full["ids"]
        row["engine_full_identical_to_hf"] = full_ids[: len(refs[key][1])] == refs[key][1]
        row["engine_full_tok_s"] = 1.0 / max(full["decode_s_per_tok"], 1e-9)

        # (b) masked streaming + verify gate
        with ResourceMonitor(interval_s=0.5, battery_every_n=20) as mon:
            masked = eng.generate(ids, args.max_new, mass=args.mass,
                                  keep=args.mass if args.mass < 1.0 else 0.9,
                                  verify_k=args.verify_k, eos_id=eos)
            peak_mb = PROC.memory_info().rss / 1048576
        trim_ws()
        time.sleep(1)
        released_mb = PROC.memory_info().rss / 1048576

        mids = masked["ids"]
        ref_ids = refs[key][1]
        same = 0
        for a, b in zip(mids, ref_ids):
            if a != b:
                break
            same += 1
        row.update({
            "hf_ref": refs[key][0],
            "engine_full_answer": tok.decode(full_ids, skip_special_tokens=True),
            "masked_answer": tok.decode(mids, skip_special_tokens=True),
            "masked_identical_prefix_pct": 100.0 * same / max(len(ref_ids), 1),
            "keep_mean": masked["keep_mean"],
            "pool_touched_mb": masked["pool_touched_mb"],
            "prefill_touched_mb": masked["prefill_touched_mb"],
            "masked_tok_s": 1.0 / max(masked["decode_s_per_tok"], 1e-9),
            "gate": masked["gate"],
            "rss_fixed_mb": fixed_mb,
            "rss_decode_peak_mb": peak_mb,
            "rss_released_mb": released_mb,
            "res_summary": mon.report().summary(),
        })
        print(f"  [{key}] identical_prefix {row['masked_identical_prefix_pct']:.0f}% | "
              f"keep {masked['keep_mean']:.0%} | pool {masked['pool_touched_mb']:.0f} MB | "
              f"{row['masked_tok_s']:.2f} tok/s | gate={masked['gate'].get('note')}", flush=True)
        print(f"      RSS fixed {fixed_mb:.0f} -> decode peak {peak_mb:.0f} -> "
              f"released {released_mb:.0f} MB", flush=True)
        results["rows"].append(row)

    # ---------------- summary ----------------------------------------------
    print("\n================ PHASE B SUMMARY ================")
    for r in results["rows"]:
        print(f"{r['key']:7s} keep {r['keep_mean']:5.1%} | prefix-identical "
              f"{r['masked_identical_prefix_pct']:5.1f}% | pool {r['pool_touched_mb']:6.0f} MB "
              f"| gate: {r['gate'].get('note', 'off')}")
    print(f"RSS: fixed {fixed_mb:.0f} MB | decode peak "
          f"{max(r['rss_decode_peak_mb'] for r in results['rows']):.0f} MB | released "
          f"{min(r['rss_released_mb'] for r in results['rows']):.0f} MB")

    out = args.out or f"results/proto_phaseB_{model_dir.name}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
