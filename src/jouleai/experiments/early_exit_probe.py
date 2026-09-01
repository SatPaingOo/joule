"""Early-exit probe: is adaptive depth control viable? (the "sense" vision)

The vision: "a question should use as much depth as it needs — easy questions
can answer at layer 12, hard ones need all 36." If per-layer exit heads give
high-confidence, correct answers early for SOME questions, then an adaptive
depth controller (converter-trained exit heads + runtime confidence gate +
fallback) is a real compute saver.

Method (train-free, uses the model's own lm_head as the exit head — no extra
training, this is the "early-exit via shared head" approximation):
  - Run the full Qwen3-8B forward; at every layer capture the hidden state
    h_l (after RMSNorm), project through the model's final lm_head (tied
    weights) -> per-layer logits.
  - For each layer: greedy argmax vs the FULL model's final answer.
  - Confidence = softmax margin (top1 - top2). A layer is "exit-worthy" if
    its argmax MATCHES the full model AND margin is high.
  - Measure per prompt: the earliest layer where the exit answer == full
    answer (with a margin threshold). Compute saved = 1 - exit_layer/L.

This is the honest test: if easy questions consistently exit at 30-60% depth
with correct answers, adaptive depth works. If exit requires near-full depth,
it doesn't.

Run: python src/jouleai/experiments/early_exit_probe.py --model models/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import sys
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
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=6)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()

    model_dir = Path(args.model)
    store = SenseWeightStore(model_dir)
    c = json.loads((model_dir / "config.json").read_text())
    d = c["hidden_size"]
    L = c["num_hidden_layers"] if not args.layers else args.layers
    n_heads, n_kv = c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim", d // n_heads)
    eps = c["rms_norm_eps"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)

    embed = store.full("model.embed_tokens.weight")
    final_norm = store.full("model.norm.weight")
    lm_head = store.full("lm_head.weight") if not c.get("tie_word_embeddings") else embed

    wq, wk, wv, wo, w1, w2, w3, n1, n2 = [], [], [], [], [], [], [], [], []
    for l in range(L):
        p = f"model.layers.{l}"
        wq.append(store.full(f"{p}.self_attn.q_proj.weight"))
        wk.append(store.full(f"{p}.self_attn.k_proj.weight"))
        wv.append(store.full(f"{p}.self_attn.v_proj.weight"))
        wo.append(store.full(f"{p}.self_attn.o_proj.weight"))
        w1.append(store.full(f"{p}.mlp.gate_proj.weight"))
        w2.append(store.full(f"{p}.mlp.up_proj.weight"))
        w3.append(store.full(f"{p}.mlp.down_proj.weight"))
        n1.append(store.full(f"{p}.input_layernorm.weight"))
        n2.append(store.full(f"{p}.post_attention_layernorm.weight"))

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    theta = c.get("rope_theta", 1e6)
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    pos = torch.arange(2048).float()[:, None]
    fr = pos * inv[None, :]
    emb = torch.cat([fr, fr], dim=-1)
    cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def rope_apply(x, p):
        half = hd // 2
        x1, x2 = x[..., :half], x[..., half:]
        cc = cos[p][..., :half].unsqueeze(1)
        ss = sin[p][..., :half].unsqueeze(1)
        return torch.cat([x1 * cc - x2 * ss, x1 * ss + x2 * cc], dim=-1)

    def forward_with_exits(input_ids):
        """Forward, returning per-layer exit logits + final logits."""
        T = input_ids.shape[0]
        x = embed[input_ids]
        exits = []
        for l in range(L):
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
            x = x + (o.reshape(T, n_heads * hd)) @ wo[l].T
            h2 = rms(x, n2[l])
            g = h2 @ w1[l].T
            up = h2 @ w2[l].T
            x = x + (F.silu(g) * up) @ w3[l].T
            # exit logits at this layer (use final-norm-style projection)
            exits.append((rms(x, final_norm) @ lm_head.T))
        return exits

    print(f"model {model_dir.name} | {L} layers | prompts={args.prompts} | "
          f"margin_threshold={args.margin}", flush=True)
    prompts = PROMPTS[: args.prompts]
    ids_list = [tok(p, return_tensors="pt").input_ids[0] for p in prompts]

    all_exit_layers = []
    print("\n== per-prompt earliest correct exit (last token) ==", flush=True)
    for pi, ids in enumerate(ids_list):
        with torch.no_grad():
            exits = forward_with_exits(ids)
        final_logits = exits[-1]
        final_ans = final_logits[-1].argmax().item()
        # find earliest layer where last-token argmax == final AND margin >= thr
        exit_layer = None
        for l in range(L):
            lg = exits[l][-1]
            top2 = torch.topk(lg, 2).values
            margin = (top2[0] - top2[1]).item()
            if lg.argmax().item() == final_ans and margin >= args.margin:
                exit_layer = l
                break
        if exit_layer is None:
            # even full-depth margin is below threshold -> "no early exit"
            for l in range(L):
                if exits[l][-1].argmax().item() == final_ans:
                    exit_layer = l
            print(f"  prompt {pi}: NO early exit (margin<{args.margin} until "
                  f"layer {exit_layer if exit_layer is not None else 'never'})", flush=True)
        else:
            saved = 1.0 - (exit_layer + 1) / L
            print(f"  prompt {pi}: exit at layer {exit_layer} "
                  f"({(exit_layer+1)/L*100:.0f}% depth, save {saved*100:.0f}%)", flush=True)
            all_exit_layers.append(exit_layer)

    # summary
    n_exit = len(all_exit_layers)
    print(f"\n== summary ==", flush=True)
    print(f"  prompts with early exit (argmax match + margin>={args.margin}): "
          f"{n_exit}/{len(ids_list)}", flush=True)
    if n_exit:
        avg = sum(all_exit_layers) / n_exit
        print(f"  avg exit layer: {avg:.1f} / {L} "
              f"= {avg/L*100:.0f}% depth (avg compute saved "
              f"{100*(1-(avg+1)/L):.0f}%)", flush=True)

    # greedy multi-token: does early-exit hold for generated tokens?
    print("\n== greedy first-token exit quality (full model vs exit at 50% depth) ==", flush=True)
    half = L // 2
    match = 0
    for pi, ids in enumerate(ids_list):
        with torch.no_grad():
            exits = forward_with_exits(ids)
        if exits[half][-1].argmax().item() == exits[-1][-1].argmax().item():
            match += 1
    print(f"  first-token argmax match at 50% depth: {match}/{len(ids_list)}", flush=True)

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
