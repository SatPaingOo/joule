"""Drift bisect: native decode-step KV vs HF bf16 at the divergence position.

Entry 72 proved prefill KV correct; this compares DECODE-step KV (positions
>= prompt_len) with native replaying the exact HF token prefix, so the two
streams share the same inputs up to the divergence token. If decode KV
diverges at pos k while the logits are still identical, the bug is in the
decode-step KV write path (not routing / not attention scale).

Usage:
    python src/jouleai/experiments/drift_bisect.py <model_dir> [--query N]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.native.decoder3 import NativeDecoder  # noqa: E402

_QUERIES = [
    "What is the capital of France? Answer in one sentence.",
    "Explain photosynthesis in simple terms.",
]


def main() -> int:
    model_dir = Path(sys.argv[1])
    qi = 0
    if "--query" in sys.argv:
        qi = int(sys.argv[sys.argv.index("--query") + 1])
    q = _QUERIES[qi]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    hf.eval()
    eos = tok.eos_token_id

    t = tok.apply_chat_template([{"role": "user", "content": q}],
                                add_generation_prompt=True, tokenize=False)
    ids = tok(t, return_tensors="pt")

    # ---- HF greedy 64 tokens (capture hidden states + KV every step) ----
    N = 64
    past = None
    hf_tokens: list[int] = []
    hf_logits = []  # per step: [V] fp32 (prediction AT that step's position)
    hf_attn = []   # per step: list of [L] attention output tensors [1,1,d]
    hf_hidden = [] # per step: list of [L+1] hidden states [1,1,d] (pre-attn res)
    with torch.no_grad():
        cur = ids["input_ids"]
        pos_abs = cur.shape[1]
        # first: prompt forward to get KV, and record per-layer hidden states
        outs = hf.model(input_ids=cur, use_cache=True, output_hidden_states=True)
        hf_hidden.append([x.float().cpu() for x in outs.hidden_states])
        past = outs.past_key_values
        lg = hf.lm_head(hf.model.norm(outs.last_hidden_state))[:, -1]
        hf_logits.append(lg.float().cpu()[0])
        nxt = int(lg.argmax())
        hf_tokens.append(nxt)
        for k in range(N - 1):
            cur = torch.tensor([[nxt]])
            outs = hf.model(input_ids=cur, past_key_values=past, use_cache=True,
                            output_hidden_states=True)
            hf_hidden.append([x.float().cpu() for x in outs.hidden_states])
            past = outs.past_key_values
            lg = hf.lm_head(hf.model.norm(outs.last_hidden_state))[:, -1]
            hf_logits.append(lg.float().cpu()[0])
            nxt = int(lg.argmax())
            hf_tokens.append(nxt)

    P = ids["input_ids"].shape[1]
    print(f"[hf] prompt_len={P} 64-token greedy:")
    print("     " + tok.decode(hf_tokens, skip_special_tokens=True)[:120])
    print("     token ids:", hf_tokens[:24], "...")

    # ---- native replay of the HF token stream (decode_spec_verify) ----
    # One batch call decodes all N positions sequentially over shared KV and
    # returns the FULL logits at each position — the per-step logit curve.
    nd = NativeDecoder(model_dir)
    nd.reset()
    t0 = time.perf_counter()
    nd.prefill(ids["input_ids"][0].tolist())
    toks = hf_tokens[:N]
    poss = list(range(P, P + N))
    lgs = nd.decode_spec_verify(toks, poss)   # [N] each [V] fp32
    wall = time.perf_counter() - t0
    print(f"[native] replay of {N} HF tokens via decode_spec_verify in {wall:.1f}s")

    print("\n=== per-step logit comparison (native vs HF) ===")
    print("  (native replay logits[b] predict token b+1 — compare vs HF logits[b+1])")
    print("  k:  HF token | native argmax | max|dlogit| | HF margin | verdict")
    div = -1
    for b in range(N - 1):
        nl = torch.from_numpy(lgs[b]).float()
        na = int(nl.argmax())
        hl = hf_logits[b + 1]                      # same absolute position
        ha = int(hl.argmax())
        d = float((nl - hl).abs().max())
        margin = float(hl[ha] - hl[na])            # how close native's choice is
        same = (na == ha)
        if div < 0 and not same:
            div = b + 1                            # HF token index that diverges
        print(f"  {b:2d}: {ha:6d} | {na:6d} | {d:9.3f} | {margin:8.3f} | "
              f"{'SAME' if same else 'DIVERGE'}")
    if div < 0:
        div = N - 1
        print("[native] 64 tokens identical to HF — no drift on this query.")
    print(f"\n[native] first divergence at decode index {div} "
          f"(abs position {P + div})")

    # ---- compare KV at the last IDENTICAL-input position (native vs HF) ----
    # Position P+div-1 is written from identical input tokens on both sides;
    # if the decode-step KV write differs there, the decode KV path is wrong.
    L = nd.cfg.L
    print(f"\n=== decode-step KV at position {P + div - 1} "
          f"(last identical-input pos, decode idx {div - 1}) ===")
    if hasattr(past, "key_cache"):
        kc, vc = past.key_cache, past.value_cache
    else:
        kc = [p[0] for p in past]
        vc = [p[1] for p in past]
    npos = P + div - 1
    for l in range(L):
        hfk, hfv = kc[l][0], vc[l][0]               # [H, T, hd] / [H, T, hd]
        nk, nv = nd._seq_kv["seq0"][l]
        # native stores one KV row per (pos, head); HF stores per (head, pos)
        hk = hfk[:, npos].float().numpy()           # [H, hd]
        hv = hfv[:, npos].float().numpy()
        nkrow = nk[npos]                            # [n_kv, hd]
        nvrow = nv[npos]
        if nkrow.shape[0] != hk.shape[0]:
            print(f"  L{l}: shape mismatch native {nkrow.shape} vs HF {hk.shape}")
            continue
        diff_k = np.abs(nkrow - hk).max()
        diff_v = np.abs(nvrow - hv).max()
        normk = np.linalg.norm(hk)
        normv = np.linalg.norm(hv)
        corr_k = np.corrcoef(nkrow.ravel(), hk.ravel())[0, 1]
        corr_v = np.corrcoef(nvrow.ravel(), hv.ravel())[0, 1]
        tag = "OK " if diff_k < 1e-2 and corr_k > 0.999 else "BAD"
        print(f"  L{l:2d}: K maxdiff={diff_k:8.4f} corr={corr_k:8.4f} "
              f"(norm {normk:7.2f}) | V maxdiff={diff_v:8.4f} corr={corr_v:8.4f} "
              f"(norm {normv:7.2f})  {tag}")

    # ---- also compare KV 3 positions earlier (sanity: drift is recent?) ----
    print(f"\n=== decode-step KV at position {P + div - 3} (3 steps before) ===")
    npos2 = P + div - 4
    for l in range(0, L, 3):
        hfk, hfv = kc[l][0], vc[l][0]
        nk, nv = nd._seq_kv["seq0"][l]
        hk = hfk[:, npos2].float().numpy()
        nkrow = nk[npos2]
        diff_k = np.abs(nkrow - hk).max()
        corr_k = np.corrcoef(nkrow.ravel(), hk.ravel())[0, 1]
        print(f"  L{l:2d}: K maxdiff={diff_k:8.4f} corr={corr_k:8.4f}")

    # ---- routing comparison at the last identical-input position ----
    print(f"\n=== MoE routing at decode position {P + div - 1} "
          f"(last identical-input pos) ===")
    print("(native routing weights vs HF — compare per layer, top-8)")
    # Re-run HF forward one more step with output_router_logits to capture
    # the routing at the same position the native replay just computed.
    with torch.no_grad():
        cur = torch.tensor([[hf_tokens[div - 1]]])
        outs = hf.model(input_ids=cur, past_key_values=past, use_cache=True,
                        output_router_logits=True)
        rlogits = [x[0].float().cpu() for x in outs.router_logits]  # [L] [E]
    # native: re-decode pos P+div-1 and grab ws->scores before softmax is lost;
    # instead compare router INPUT (post-norm2 hidden) — see kernel ws->scores.
    print("  (native router logits are in the kernel workspace — computed "
          "inside layer_ffn_batch; see per-layer hidden comparison next)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
