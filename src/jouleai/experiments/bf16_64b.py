"""Definitive bf16-expert 64-token test vs HF (clean per-step references)."""
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
P = int(ids["input_ids"].shape[1])

# HF 64-token greedy (single generate call, transformers-managed)
with torch.no_grad():
    o = hf.generate(**ids, max_new_tokens=64, do_sample=False,
                    pad_token_id=tk.eos_token_id)
    hf_tokens = o[0, ids["input_ids"].shape[1]:].tolist()
print("HF 64:", hf_tokens[:24])

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
cache = {}
t0 = time.perf_counter()
lg = eng.forward(ids, cache, 0, pool)
out = [int(lg[0, -1].argmax())]
for i in range(63):
    step = torch.tensor([[out[-1]]])
    lg = eng.forward(step, cache, P + i, pool)
    out.append(int(lg[0, -1].argmax()))
wall = time.perf_counter() - t0
print(f"bf16 py: {len(out)} tokens in {wall:.1f}s")
print("bf16 py:", out[:24])
mism = [i for i in range(min(len(out), len(hf_tokens))) if out[i] != hf_tokens[i]]
print("bf16 py vs HF mismatches:", mism[:16])
