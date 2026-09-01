"""Verify the bf16-expert Python path is 64-token identical to HF.

The GenericStreamer with full bf16 experts (InlinePool) was verified to have
correct math; this confirms the full 64-step greedy matches HF token-for-token.
"""
import sys
sys.path.insert(0, "src")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

MD = "models/OLMoE-1B-7B-0824-Instruct"
tk = AutoTokenizer.from_pretrained(MD)
hf = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16)
hf.eval()
q = "What is the capital of France? Answer in one sentence."
t = tk.apply_chat_template([{"role": "user", "content": q}],
                           add_generation_prompt=True, tokenize=False)
ids = tk(t, return_tensors="pt")

# HF 64-token greedy
with torch.no_grad():
    o = hf.generate(**ids, max_new_tokens=64, do_sample=False,
                    pad_token_id=tk.eos_token_id)
    hf_tokens = o[0, ids["input_ids"].shape[1]:].tolist()
print("HF 64:", hf_tokens[:20], "...")

from jouleai.engine.generic_streamer import GenericStreamer
from jouleai.storage.weight_store import SenseWeightStore

eng = GenericStreamer(MD)

class InlinePool:
    def __init__(self, store):
        self.store = store
    def ensure(self, l, e):
        p = f"model.layers.{l}.mlp.experts.{e}"
        return tuple(self.store.full(f"{p}.{x}_proj.weight")
                     for x in ("gate", "up", "down"))

pool = InlinePool(eng.store)
eos = tk.eos_token_id
t0 = time.perf_counter()
g = eng.generate(ids, 64, pool, eos)
wall = time.perf_counter() - t0
print(f"bf16 py: {len(g['ids'])} tokens in {wall:.1f}s")
mism = [i for i in range(min(len(g['ids']), len(hf_tokens)))
        if g['ids'][i] != hf_tokens[i]]
print("bf16 py vs HF mismatches:", mism[:12])
print("bf16 py:", g['ids'][:20])
