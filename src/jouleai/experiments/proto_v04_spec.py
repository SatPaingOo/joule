"""Step 3: speculative decoding — Qwen2.5-1.5B drafts, Qwen3-30B-MoE verifies.

Greedy spec decode is EXACT: output equals the target model's own greedy
output, but the target runs once per ~gamma draft tokens instead of once per
token. Measured: effective tok/s, acceptance rate, identity vs pure greedy.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.experiments.proto_v04_qwen3moe import Qwen3Streamer  # noqa: E402
from jouleai.native.moe import NativeMoE  # noqa: E402
from jouleai.storage.q4_store import Q4ExpertPool  # noqa: E402

QUERIES = {
    "easy": "What is the capital of France? Answer in one sentence.",
    "medium": "Explain photosynthesis in simple terms, in about 3 sentences.",
    "hard": (
        "A train travels 120 km in 90 minutes. Its speed then halves. "
        "How long for another 60 km?"
    ),
}


class DraftModel:
    """Qwen2.5-1.5B greedy drafter with a croppable KV cache."""

    def __init__(self, path: str):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        self.model.eval()
        self.cache = DynamicCache()
        self.len = 0  # tokens fed so far

    def prompt(self, messages):
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=False)
        ids = self.tok(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**ids, past_key_values=self.cache, use_cache=True)
        self.p_len = ids["input_ids"].shape[1]
        self.len = self.p_len
        return int(out.logits[0, -1].argmax())

    def extend_and_draft(self, feed: list[int], gamma: int) -> list[int]:
        """Feed accepted tokens, then draft gamma greedy tokens."""
        if feed:
            ids = torch.tensor([feed])
            with torch.no_grad():
                out = self.model(ids, past_key_values=self.cache, use_cache=True)
            self.len += len(feed)
            last = out.logits[0, -1]
        else:
            return []
        drafted = [int(last.argmax())]
        cur = torch.tensor([[drafted[0]]])
        for _ in range(gamma - 1):
            with torch.no_grad():
                out = self.model(cur, past_key_values=self.cache, use_cache=True)
            self.len += 1
            nxt = int(out.logits[0, -1].argmax())
            drafted.append(nxt)
            cur = torch.tensor([[nxt]])
        return drafted

    def crop(self, keep: int):
        self.cache.crop(keep)
        self.len = keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="models/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--draft", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--budget-gb", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    md = Path(args.target)
    tok = AutoTokenizer.from_pretrained(md)
    eos = tok.eos_token_id
    eng = Qwen3Streamer(md)
    torch.set_num_threads(4)
    from concurrent.futures import ThreadPoolExecutor
    eng._executor = ThreadPoolExecutor(max_workers=8)
    eng._native_moe = NativeMoE(eng.d, eng.cfg["moe_intermediate_size"], k=8)
    pool = Q4ExpertPool(md, eng.n_layers, eng.cfg["num_experts"],
                        int(args.budget_gb * 1073741824), raw=True)

    print("loading draft model ...", flush=True)
    draft = DraftModel(args.draft)

    def prompt_ids(q):
        text = tok.apply_chat_template([{"role": "user", "content": q}],
                                       add_generation_prompt=True, tokenize=False)
        return tok(text, return_tensors="pt")

    results = {"gamma": args.gamma, "rows": []}

    # baseline: pure target greedy (for identity check), one query only (slow)
    base_q = "easy"
    t0 = time.perf_counter()
    base = eng.generate(prompt_ids(QUERIES[base_q]), args.max_new, pool, eos)
    base_time = time.perf_counter() - t0
    base_ids = base["ids"]
    print(f"[baseline pure target] {base_time:.0f}s, {base['tok_s']:.2f} tok/s",
          flush=True)

    for key, q in QUERIES.items():
        ids = prompt_ids(q)
        # ---- target prefill: cache ends at prompt; first token chosen ----
        t0 = time.perf_counter()
        cache: dict = {}
        with torch.no_grad():
            logits = eng.forward(ids, cache, 0, pool)
        p_len = ids["input_ids"].shape[1]
        emitted = [int(logits[0, -1].argmax())]   # pending: not yet fed to target
        prefill_s = time.perf_counter() - t0

        # ---- draft prompt prefill ----
        t_draft0 = time.perf_counter()
        draft.prompt([{"role": "user", "content": q}])
        draft_prefill_s = time.perf_counter() - t_draft0

        pos = p_len                      # target cache end (== prompt)
        last_tok = emitted[0]            # pending token to feed next round
        accept_total, rounds = 0, 0
        t0 = time.perf_counter()
        while len(emitted) < args.max_new:
            # draft: feed pending token, then gamma drafts
            d_base = draft.len
            drafted = draft.extend_and_draft([last_tok], args.gamma)
            # verify: feed [last_tok] + drafted[:-1] -> gamma judging logits
            verify_tokens = [last_tok] + drafted[:-1]
            step = torch.tensor([verify_tokens])
            with torch.no_grad():
                vlogits = eng.forward(step, cache, pos, pool)[0]  # [gamma, V]
            pos += len(verify_tokens)
            accept = 0
            rejected = False
            for j in range(args.gamma):
                nxt = int(vlogits[j].argmax())
                if nxt == drafted[j]:
                    accept += 1
                    emitted.append(drafted[j])
                else:
                    emitted.append(nxt)
                    rejected = True
                    break
            if not rejected and len(emitted) < args.max_new:
                emitted.append(int(vlogits[args.gamma - 1].argmax()))
            accept_total += accept
            rounds += 1
            # trim target cache to prompt + emitted[:-1] (last token stays pending)
            keep_t = p_len + len(emitted) - 1
            for l2 in list(cache):
                kk, vv = cache[l2]
                cache[l2] = (kk[:, :, :keep_t, :], vv[:, :, :keep_t, :])
            pos = keep_t
            # draft cache: keep prompt + emitted[:-1] relative to its own prompt len
            draft.crop(draft.p_len + len(emitted) - 1)
            last_tok = emitted[-1]
        wall = time.perf_counter() - t0
        n = len(emitted)
        row = {
            "key": key, "answer": tok.decode(emitted, skip_special_tokens=True),
            "decode_tok_s": round(n / wall, 2),
            "prefill_s": round(prefill_s, 1),
            "draft_prefill_s": round(draft_prefill_s, 1),
            "rounds": rounds, "accepted": accept_total,
            "acceptance": round(accept_total / max(rounds * args.gamma, 1), 2),
            "tokens": n,
        }
        results["rows"].append(row)
        print(f"[{key}] {row['decode_tok_s']} tok/s decode | acceptance "
              f"{row['acceptance']} | rounds {rounds} | prefill {row['prefill_s']}s",
              flush=True)
        print(f"   {row['answer'][:80]!r}", flush=True)
        if key == base_q:
            row["identical_to_baseline"] = emitted == base_ids
            print(f"   identical to pure-greedy baseline: {row['identical_to_baseline']}")

    out = args.out or "results/proto_spec_decode.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
