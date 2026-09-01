"""Definitive test: native bf16-expert 64-token generation vs HF.

The bf16 expert tier removes quantization entirely — the FFN reads exact bf16
rows. This is the fix for the 64-token drift (q4/i8 accumulate to a logit
flip at step 7; bf16 should stay token-identical).
"""
import sys
import time
sys.path.insert(0, "src")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MD = "models/OLMoE-1B-7B-0824-Instruct"
tk = AutoTokenizer.from_pretrained(MD)
hf = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16)
hf.eval()
q = "What is the capital of France? Answer in one sentence."
t = tk.apply_chat_template([{"role": "user", "content": q}],
                           add_generation_prompt=True, tokenize=False)
ids = tk(t, return_tensors="pt")
P = int(ids["input_ids"].shape[1])

with torch.no_grad():
    o = hf.generate(**ids, max_new_tokens=64, do_sample=False,
                    pad_token_id=tk.eos_token_id)
    hf_tokens = o[0, ids["input_ids"].shape[1]:].tolist()
print("HF 64:", hf_tokens[:24])

from jouleai.native.decoder3 import NativeDecoder
nd = NativeDecoder(MD, precision="bf16")
nd.reset()
t0 = time.perf_counter()
lg = nd.prefill(ids["input_ids"][0].tolist())
out = [int(lg.argmax())]
for i in range(63):
    if out[-1] == tk.eos_token_id:
        break
    out.append(nd.decode_token(out[-1]))
wall = time.perf_counter() - t0
print(f"bf16 native: {len(out)} tokens in {wall:.1f}s ({len(out)/wall:.1f} tok/s)")
print("bf16 native:", out[:24])
mism = [i for i in range(min(len(out), len(hf_tokens))) if out[i] != hf_tokens[i]]
print("bf16 native vs HF mismatches:", mism[:16])
