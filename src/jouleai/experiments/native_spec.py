"""Native speculative decoding: Qwen3-0.6B drafts, the native C target
(Qwen3-30B-A3B) verifies gamma drafts in ONE decode_spec_verify call.

Exact: the spec output equals the target's own greedy (the verify passes the
draft tokens through the target itself), but the target runs once per gamma
drafts with weight reads amortized (shared-KV batch — the batch kernel's
strength).

Correctness of the verify (fixed for Entry 69): the verify batch is
[last_real, d_1, ..., d_{g-1}] at positions base..base+g-1 over a SHARED KV
(seq0). The kernel processes seqs in order per layer, so each draft token is
verified against prefix + the previously-accepted draft tokens — the correct
autoregressive semantics. The Entry 27/54 harness verified each draft against
prefix-only KV (per-seq separate buffers) — wrong, acceptance was meaningless.

RESULT (Entry 69): the shared-KV verify is CORRECT (batch logits == sequential
at argmax level; first position bit-identical 0.0) — but acceptance with the
Qwen3-0.6B draft is ~0: the 0.6B is 50x smaller than the 30B target and its
first drafted token almost never matches the target's greedy ("of" vs
"France"), so the longest-prefix loop rejects at k=0 every round and the
following drafts (which DO match) are never used. Confirms Entry 27: a
~50x-size draft gap kills spec decode. A closer draft (Qwen3-1.7B/4B, when
available) is the fix; the verify machinery here is ready.

Run: python src/jouleai/experiments/native_spec.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.native.decoder3 import NativeDecoder  # noqa: E402

QUERIES = {
    "easy": "What is the capital of France? Answer in one sentence.",
    "medium": "Explain photosynthesis in simple terms, about 3 sentences.",
}


class Qwen3Draft:
    """Qwen3-0.6B (GGUF, bf16) drafter; tokenizer shared with the target so
    token ids are directly comparable. KV croppable (rejected drafts removed)."""

    def __init__(self, gguf_dir: str, gguf_file: str, target_tok):
        from transformers import AutoModelForCausalLM
        from transformers.cache_utils import DynamicCache
        self.tok = target_tok
        self._DynamicCache = DynamicCache
        self.model = AutoModelForCausalLM.from_pretrained(
            gguf_dir, gguf_file=gguf_file, dtype=torch.bfloat16)
        self.model.eval()
        self.kv = DynamicCache()
        self.len = 0
        self.p_len = 0

    def prefill(self, ids: list[int]):
        self.p_len = len(ids)
        self.kv = self._DynamicCache()
        with torch.no_grad():
            self.model(input_ids=torch.tensor([ids]), past_key_values=self.kv,
                       use_cache=True)
        self.len = len(ids)

    def extend_and_draft(self, feed: list[int], gamma: int) -> list[int]:
        """Feed the last token, then greedy-draft gamma more tokens."""
        out = []
        cur = feed[:]
        with torch.no_grad():
            for _ in range(gamma + 1):
                logits = self.model(input_ids=torch.tensor([cur]),
                                    past_key_values=self.kv, use_cache=True).logits
                nxt = int(logits[0, -1].argmax())
                out.append(nxt)
                cur = [nxt]
        self.len += len(feed)
        return out[1:]  # gamma drafted tokens (feed token already consumed)

    def crop(self, keep: int):
        self.kv.crop(keep)
        self.len = keep


def greedy_reference(nd, ids: list[int], max_new: int, eos) -> list[int]:
    """Plain greedy decode through the native kernel (the exact baseline)."""
    nd.reset()
    logits = nd.prefill(ids)
    out = [int(logits.argmax())]
    for _ in range(max_new - 1):
        if out[-1] == eos:
            break
        out.append(nd.decode_token(out[-1]))
    return out


def spec_decode(nd, draft, ids: list[int], gamma: int, max_new: int,
                eos) -> list[int]:
    """Spec decode: draft gamma, verify in ONE shared-KV batch call, accept
    the longest prefix the target agrees with (exact = greedy)."""
    logits = nd.prefill(ids)
    p_len = len(ids)
    emitted = [int(logits.argmax())]
    draft.prefill(ids)
    accept_total, rounds = 0, 0
    while len(emitted) < max_new:
        if emitted[-1] == eos:
            break
        base = p_len + len(emitted) - 1           # position of emitted[-1]
        drafted = draft.extend_and_draft([emitted[-1]], gamma)  # d_1..d_g
        # verify: [last_real, d_1..d_{g-1}] at positions base..base+g-1,
        # shared KV (seq0) — d_j checked against prefix + accepted drafts.
        verify_toks = [emitted[-1]] + drafted[:-1]
        positions = [base + j for j in range(gamma)]
        logits_v = nd.decode_spec_verify(verify_toks, positions)
        k = 0
        while k < gamma - 1 and int(logits_v[k].argmax()) == drafted[k]:
            k += 1                                # d_1..d_k accepted
        accepted = drafted[:k]
        nxt = int(logits_v[k].argmax())           # the target's own choice
        emitted.extend(accepted + [nxt])
        accept_total += k
        rounds += 1
        # trim the draft cache back to the last real token (drop rejected)
        draft.crop(draft.p_len + len(emitted) - 1)
    return emitted, accept_total, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--draft-dir", default="models/Qwen3-0.6B-GGUF")
    ap.add_argument("--draft-file", default="Qwen3-0.6B-BF16.gguf")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    eos = tok.eos_token_id
    nd = NativeDecoder(args.model, max_tokens=512)
    draft = Qwen3Draft(args.draft_dir, args.draft_file, tok)
    print(f"target {args.model} | draft Qwen3-0.6B (bf16) | gamma {args.gamma}",
          flush=True)

    for key, q in QUERIES.items():
        ids = tok(q, return_tensors="pt").input_ids[0].tolist()
        # exact baseline
        ref = greedy_reference(nd, list(ids), args.max_new, eos)
        # spec decode
        t0 = time.perf_counter()
        spec, acc, rounds = spec_decode(nd, draft, list(ids), args.gamma,
                                        args.max_new, eos)
        wall = time.perf_counter() - t0
        ident = spec[:len(ref)] == ref
        n = len(spec)
        rate = (n - 1) / wall
        acc_rate = acc / max(rounds * (args.gamma - 1), 1)
        print(f"[{key}] spec {rate:.1f} tok/s | acceptance {acc_rate:.2f} | "
              f"{rounds} rounds | exact-vs-greedy {ident}", flush=True)
        print(f"  answer: {tok.decode(spec, skip_special_tokens=True)[:90]}",
              flush=True)
        if not ident:
            for i, (a, b) in enumerate(zip(spec, ref)):
                if a != b:
                    print(f"  DIVERGE at {i}: spec {tok.decode([a])!r} "
                          f"vs greedy {tok.decode([b])!r}", flush=True)
                    break


if __name__ == "__main__":
    main()
