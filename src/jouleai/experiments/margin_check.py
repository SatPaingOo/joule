"""Check whether the step-7/12/14 divergences are near-ties (HF margin)."""
import sys
sys.path.insert(0, "src")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

MD = "models/OLMoE-1B-7B-0824-Instruct"
tk = AutoTokenizer.from_pretrained(MD)
hf = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16)
hf.eval()
q = "What is the capital of France? Answer in one sentence."
t = tk.apply_chat_template([{"role": "user", "content": q}],
                           add_generation_prompt=True, tokenize=False)
ids = tk(t, return_tensors="pt")
P = int(ids["input_ids"].shape[1])

# HF: 64-step greedy with full logits captured per position
hf_lg = []
past = None
with torch.no_grad():
    outs = hf.model(input_ids=ids["input_ids"], use_cache=True)
    past = outs.past_key_values
    lg = hf.lm_head(hf.model.norm(outs.last_hidden_state))[:, -1]
    hf_lg.append(lg.float().cpu()[0])
    for _ in range(63):
        cur = torch.tensor([[int(lg.argmax())]])
        outs = hf.model(input_ids=cur, past_key_values=past, use_cache=True)
        past = outs.past_key_values
        lg = hf.lm_head(hf.model.norm(outs.last_hidden_state))[:, -1]
        hf_lg.append(lg.float().cpu()[0])
hf_tokens = [int(l.argmax()) for l in hf_lg]

from jouleai.native.decoder3 import NativeDecoder
for prec in ("q4", "i8"):
    nd = NativeDecoder(MD, precision=prec)
    nd.reset()
    nd.prefill(ids["input_ids"][0].tolist())
    seq = []
    for j in range(63):
        nxt = nd.decode_token(hf_tokens[j])
        seq.append(nxt)
    print(f"=== {prec} B=1: 64-step replay vs HF ===")
    mism = [b for b in range(63) if seq[b] != hf_tokens[b + 1]]
    print(f"mismatches at steps: {mism[:12]}")
    for b in mism[:6]:
        hl = hf_lg[b + 1]
        margin = float(hl[hl.argmax()] - hl[seq[b]])
        nl = torch.tensor(seq[b])
        # how far is the native choice under HF's own distribution
        print(f"  step {b}: native={seq[b]} HF={hf_tokens[b+1]} "
              f"HF-margin(native)={margin:.3f}")
    del nd
    gc.collect()
