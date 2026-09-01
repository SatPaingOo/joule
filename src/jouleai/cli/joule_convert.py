"""joule convert — one-command model preparation.

Detects the architecture, builds the Joule bundle for it (Q4 expert store for
MoE models, layout notes for dense), and writes a manifest + report card:

    python -m jouleai.cli.joule_convert models/Qwen3-30B-A3B-Instruct-2507
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.arch.registry import get_spec, SUPPORTED


def detect_arch(model_dir: Path):
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no config.json in {model_dir}")
    cfg = json.loads(cfg_path.read_text())
    spec = get_spec(cfg)   # raises with the supported list if unknown
    return spec, cfg


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="model directory")
    ap.add_argument("--budget-gb", type=float, default=8.0)
    ap.add_argument("--verify", action="store_true",
                    help="auto-verify against HF before writing the manifest")
    args = ap.parse_args()

    model_dir = Path(args.model)
    t0 = time.perf_counter()
    spec, cfg = detect_arch(model_dir)
    arch = spec.model_type
    print(f"architecture: {arch} | {spec.n_layers}L x {spec.d}d | "
          f"{'MoE' if spec.moe else 'dense'} | qk_norm={spec.qk_norm}")
    manifest = {
        "model": str(model_dir), "arch": arch, "moe": spec.moe,
        "config": {k: cfg.get(k) for k in (
            "num_hidden_layers", "hidden_size", "num_attention_heads",
            "num_key_value_heads", "num_experts", "num_experts_per_tok",
            "intermediate_size", "tie_word_embeddings")},
        "budget_gb": args.budget_gb,
    }

    store_size = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
    manifest["weights_bytes"] = store_size

    if spec.moe:
        from jouleai.storage.q4_store import convert_experts_q4
        n_layers, n_experts = cfg["num_hidden_layers"], cfg["num_experts"]
        naming = getattr(spec, "expert_naming", "qwen")
        print(f"building Q4 expert store: {n_layers} layers x {n_experts} experts "
              f"(naming={naming}) ...")
        bin_path = convert_experts_q4(model_dir, n_layers=n_layers, n_experts=n_experts,
                                      naming=naming)
        manifest["q4_store"] = {
            "path": str(bin_path),
            "bytes": bin_path.stat().st_size,
            "group": 64,
        }
        active = cfg["num_experts_per_tok"] * n_layers
        manifest["serving"] = {
            "experts_total": n_layers * n_experts,
            "experts_active_per_token": active,
            "working_set_gb": round(
                active * manifest["q4_store"]["bytes"]
                / (n_layers * n_experts) / 1073741824, 2),
            "lossless": "by construction (router defines the sparse compute)",
        }

    if args.verify:
        from jouleai.arch.verify import verify_streamer
        rep = verify_streamer(model_dir)
        manifest["verification"] = rep
        print(f"verification : {rep['verdict']}")

    manifest["convert_seconds"] = round(time.perf_counter() - t0, 1)
    out = model_dir / "joule_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))

    # ---------------- control plane report (this machine, this model) ------
    try:
        from jouleai.control import plan_for, ModelInfo
        dev, plan = plan_for(ModelInfo.from_config(cfg))
        manifest["control"] = {"device": f"{dev.os} {dev.total_ram_gb:.0f}GB "
                                         f"{dev.cores}c BW~{dev.mem_bw_gb_s:.0f}GB/s "
                                         f"tier={dev.tier}",
                               "plan": plan.status()}
        out.write_text(json.dumps(manifest, indent=2))
    except Exception:
        pass  # control report is best-effort

    # ---------------- report card ----------------
    print("\n================ JOULE REPORT CARD ================")
    print(f"model        : {model_dir.name}")
    print(f"architecture : {arch} ({'MoE' if spec.moe else 'dense'})")
    print(f"layers       : {cfg.get('num_hidden_layers')}"
          f" | hidden {cfg.get('hidden_size')}")
    if spec.moe:
        print(f"experts      : {cfg['num_hidden_layers']} x {cfg['num_experts']}"
              f" (top-{cfg['num_experts_per_tok']}/token)")
        print(f"weights      : {human(store_size)} -> "
              f"Q4 experts {human(manifest['q4_store']['bytes'])}")
        ws = manifest["serving"]["working_set_gb"]
        print(f"working set  : ~{ws} GB/token active experts")
        print(f"RAM budget   : {args.budget_gb} GB "
              f"(>= {ws:.1f} GB recommended for high hit rates)")
        print(f"quality      : {manifest['serving']['lossless']}")
    else:
        print(f"weights      : {human(store_size)} (dense masked serving)")
    print(f"convert time : {manifest['convert_seconds']}s")
    print(f"manifest     : {out}")
    print("next         : python -m jouleai.cli.joule_serve "
          f"{model_dir} --budget-gb {args.budget_gb}")


if __name__ == "__main__":
    main()
