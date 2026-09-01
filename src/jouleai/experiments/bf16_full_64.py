"""Full 64-token bf16 native vs HF (HF from manual 64-step loop, no EOS stop)."""
import sys, time
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

# HF 64-step greedy (manual loop, no EOS stop — the true continuation)
hf_tokens = []
past = None
with torch.no_grad():
    outs = hf.model(input_ids=ids["input_ids"], use_cache=True)
    past = outs.past_key_values
    lg = hf.lm_head(hf.model.norm(outs.last_hidden_state))[:, -1]
    hf_tokens.append(int(lg.argmax()))
    for _ in range(63):
        cur = torch.tensor([[hf_tokens[-1]]])
        outs = hf.model(input_ids=cur, past_key_values=past, use_cache=True)
        past = outs.past_key_values
        lg = hf.lm_head(hf.model.norm(outs.last_hidden_state))[:, -1]
        hf_tokens.append(int(lg.argmax()))

from jouleai.native.decoder3 import NativeDecoder
nd = NativeDecoder(MD, precision="bf16")
nd.reset()
t0 = time.perf_counter()
lg = nd.prefill(ids["input_ids"][0].tolist())
out = [int(lg.argmax())]
for i in range(63):
    out.append(nd.decode_token(out[-1]))
wall = time.perf_counter() - t0
print(f"bf16 native: 64 tokens in {wall:.1f}s ({64/wall:.1f} tok/s)")
mism = [i for i in range(64) if out[i] != hf_tokens[i]]
print("bf16 native vs HF mismatches:", mism)
print("VERDICT:", "64-TOKEN IDENTICAL" if not mism else "drifts")
print("native:", out[:20])
print("HF    :", hf_tokens[:20])
