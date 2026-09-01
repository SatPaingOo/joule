"""Layer-skip probe: which layers are skip-safe? (per-input selective passing)

Goal: for the "compute only what the input needs" vision, we need to know:
  1. Block influence per layer: how much does layer l change the hidden state?
     (skip value = ||h_after - h_before|| — a low-influence layer can be
      skipped with little drift)
  2. Output drift when skipping individual layers (logits/argmax/greedy).
  3. Is influence INPUT-DEPENDENT? (do easy queries skip more than hard ones?
     -> if yes, a per-input probe is feasible)

Method (train-free, one forward on real prompts):
  - run the full model, capture h_l before/after each layer (block influence)
  - run the model with layer l replaced by identity, compare final logits vs full
  - aggregate across a few diverse prompts

Run: python src/jouleai/experiments/layer_skip_probe.py
     --model models/Qwen3-8B [--layers 36]
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

from jouleai.storage.weight_store import SenseWeightStore  # noqa: E402


PROMPTS = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Write a haiku about the ocean.",
    "If I have 17 apples and give 5 to a friend, how many do I have left?",
    "List three reasons why exercise is good for you.",
    "Translate 'hello, how are you?' into French.",
    "What is 23 times 47?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-8B")
    ap.add_argument("--layers", type=int, default=0)  # 0 = all
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--top-prompts", type=int, default=4)
    args = ap.parse_args()

    model_dir = Path(args.model)
    store = SenseWeightStore(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    d = cfg["hidden_size"]
    L = cfg["num_hidden_layers"] if not args.layers else args.layers
    n_heads, n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim", d // n_heads)
    eps = cfg["rms_norm_eps"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)

    embed = store.full("model.embed_tokens.weight")
    final_norm = store.full("model.norm.weight")
    lm_head = store.full("lm_head.weight") if not cfg.get("tie_word_embeddings") else embed

    # load all layer weights
    wq, wk, wv, wo, w1, w2, w3, n1, n2 = [], [], [], [], [], [], [], [], []
    for l in range(L):
        p = f"model.layers.{l}"
        wq.append(store.full(f"{p}.self_attn.q_proj.weight"))
        wk.append(store.full(f"{p}.self_attn.k_proj.weight"))
        wv.append(store.full(f"{p}.self_attn.v_proj.weight"))
        wo.append(store.full(f"{p}.self_attn.o_proj.weight"))
        # Qwen3 dense: gate/up/down
        w1.append(store.full(f"{p}.mlp.gate_proj.weight"))
        w2.append(store.full(f"{p}.mlp.up_proj.weight"))
        w3.append(store.full(f"{p}.mlp.down_proj.weight"))
        n1.append(store.full(f"{p}.input_layernorm.weight"))
        n2.append(store.full(f"{p}.post_attention_layernorm.weight"))

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    # rope
    theta = cfg.get("rope_theta", 1e6)
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    pos = torch.arange(2048).float()[:, None]
    fr = pos * inv[None, :]
    emb = torch.cat([fr, fr], dim=-1)
    cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def rope_apply(x, p):
        # x: [T, n_heads, hd]; p: [T] positions; cos/sin: [maxT, hd]
        half = hd // 2
        x1, x2 = x[..., :half], x[..., half:]
        c = cos[p][..., :half].unsqueeze(1)   # [T, 1, half]
        s = sin[p][..., :half].unsqueeze(1)
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

    def attn(l, h, k_cache, v_cache, pos):
        q = (h @ wq[l].T).view(-1, n_heads, hd)
        k = (h @ wk[l].T).view(-1, n_kv, hd)
        v = (h @ wv[l].T).view(-1, n_kv, hd)
        q = rope_apply(q, pos)
        k = rope_apply(k, pos)
        rep = n_heads // n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return (o.reshape(-1, d)) @ wo[l].T

    def run(input_ids, skip: set[int] | None = None, capture_influence: bool = False):
        """Full forward; skip=set of layers to replace with identity.
        Returns logits, and (if capture) per-layer influence norms."""
        skip = skip or set()
        x = embed[input_ids]  # [T, d]
        T = input_ids.shape[0]
        k_cache = torch.zeros(L, T, n_kv, hd, dtype=torch.bfloat16)
        v_cache = torch.zeros(L, T, n_kv, hd, dtype=torch.bfloat16)
        inf = [] if capture_influence else None
        for l in range(L):
            h_in = x.clone()
            if l not in skip:
                h2 = rms(x, n1[l])
                q = (h2 @ wq[l].T).view(T, n_heads, hd)
                k = (h2 @ wk[l].T).view(T, n_kv, hd)
                v = (h2 @ wv[l].T).view(T, n_kv, hd)
                q = rope_apply(q, torch.arange(T))
                k = rope_apply(k, torch.arange(T))
                rep = n_heads // n_kv
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                x = x + (o.reshape(T, d)) @ wo[l].T
                h2 = rms(x, n2[l])
                g = h2 @ w1[l].T
                up = h2 @ w2[l].T
                x = x + (F.silu(g) * up) @ w3[l].T
            if capture_influence:
                inf.append(float((x - h_in).pow(2).mean().sqrt()))
        x = rms(x, final_norm)
        return x @ lm_head.T, inf

    # ---- measure per-layer block influence + single-layer-skip drift ----
    print(f"model {model_dir.name} | {L} layers | d={d} | prompts={args.top_prompts}", flush=True)
    prompts = PROMPTS[: args.top_prompts]
    ids_list = [tok(p, return_tensors="pt").input_ids[0] for p in prompts]

    # 1. block influence per layer (averaged over prompts)
    inf_sum = torch.zeros(L)
    for ids in ids_list:
        with torch.no_grad():
            _, inf = run(ids, capture_influence=True)
        for l, v in enumerate(inf):
            inf_sum[l] += v
    inf_avg = inf_sum / len(ids_list)
    print("\n== per-layer block influence (avg ||h_out - h_in|| over prompts) ==", flush=True)
    for l in range(L):
        print(f"  layer {l:>2}: {inf_avg[l]:.4f}", flush=True)

    # 2. skip each layer individually -> final logits drift (first prompt)
    ids = ids_list[0]
    with torch.no_grad():
        logits_full, _ = run(ids)
    drift = []
    for l in range(L):
        with torch.no_grad():
            logits_skip, _ = run(ids, skip={l})
        d_logit = float((logits_skip - logits_full).abs().max())
        drift.append(d_logit)
    print("\n== single-layer skip drift (max|dlogit|, first prompt) ==", flush=True)
    low = [l for l in range(L) if drift[l] < 0.5]
    med = [l for l in range(L) if 0.5 <= drift[l] < 2.0]
    high = [l for l in range(L) if drift[l] >= 2.0]
    print(f"  low-drift (<0.5): {low}", flush=True)
    print(f"  med-drift (0.5-2): {med}", flush=True)
    print(f"  high-drift (>=2): {high}", flush=True)

    # 3. per-prompt influence variance: is influence input-dependent?
    print("\n== input-dependence: per-layer influence per prompt ==", flush=True)
    for pi, ids in enumerate(ids_list):
        with torch.no_grad():
            _, inf = run(ids, capture_influence=True)
        top = sorted(range(L), key=lambda l: -inf[l])[:5]
        bot = sorted(range(L), key=lambda l: inf[l])[:5]
        print(f"  prompt {pi}: top-influence {top} | low-influence {bot}", flush=True)

    # 4. greedy argmax identity when skipping the lowest-influence layers
    print("\n== greedy identity: skip lowest-influence layers ==", flush=True)
    order = sorted(range(L), key=lambda l: inf_avg[l].item())
    for frac in (0.1, 0.25, 0.4):
        skip_set = set(order[: int(L * frac)])
        ident = 0
        for ids in ids_list:
            with torch.no_grad():
                full, _ = run(ids)
                skip, _ = run(ids, skip=skip_set)
            # greedy first token
            if full[0].argmax().item() == skip[0].argmax().item():
                ident += 1
        print(f"  skip top-{frac*100:.0f}% low-influence layers: "
              f"first-token argmax identical {ident}/{len(ids_list)}", flush=True)

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
