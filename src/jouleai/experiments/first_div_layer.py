"""First-divergence layer hunt at step 15 (bf16): per-layer residual, comparing
native (fp32-residual) vs HF (bf16-residual) — the residual accumulates bf16
rounding on the HF side. If the native fp32-residual is MORE precise than HF's
bf16, the native should be CLOSER to the true fp32 answer, but HF is the
reference — so native-vs-HF must converge to the bf16-rounding difference.
"""
import sys
sys.path.insert(0, "src")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

MD = "models/OLMoE-1B-7B-0824-Instruct"
tk = AutoTokenizer.from_pretrained(MD)
hf = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16)
hf.eval()
q = "What is the capital of France? Answer in one sentence."
t = tk.apply_chat_template([{"role": "user", "content": q}],
                           add_generation_prompt=True, tokenize=False)
ids = tk(t, return_tensors="pt")
P = int(ids["input_ids"].shape[1])

hf_tokens = []
past = None
with torch.no_grad():
    outs = hf.model(input_ids=ids["input_ids"], use_cache=True)
    past = outs.past_key_values
    lg = hf.lm_head(hf.model.norm(outs.last_hidden_state))[:, -1]
    hf_tokens.append(int(lg.argmax()))
    for _ in range(15):
        cur = torch.tensor([[hf_tokens[-1]]])
        outs = hf.model(input_ids=cur, past_key_values=past, use_cache=True)
        past = outs.past_key_values
        lg = hf.lm_head(hf.model.norm(outs.last_hidden_state))[:, -1]
        hf_tokens.append(int(lg.argmax()))

# HF per-layer residuals at step 15 via hooks (correct: h_after_layer)
caps = {}
pref = torch.cat([ids["input_ids"], torch.tensor([hf_tokens[:15]])], dim=1)
for l in range(16):
    caps[l] = {}
    def post_layer(mod, args, kwargs, out, l=l):
        caps[l]["h_out"] = out[0, -1].float().clone()
    hf.model.layers[l].register_forward_hook(post_layer, with_kwargs=True)
with torch.no_grad():
    outs = hf.model(input_ids=pref, use_cache=True, output_hidden_states=True)
    hs = [x[0, -1].float() for x in outs.hidden_states]

# Native bf16 per-layer residuals at step 15
from jouleai.native.decoder3 import NativeDecoder
nd = NativeDecoder(MD, precision="bf16")
nd.reset()
nd.prefill(ids["input_ids"][0].tolist())
for j in range(15):
    nd.decode_token(hf_tokens[j])
ncomp = nd.debug_decode_layers(P + 14, hf_tokens[14])
d = nd.cfg.d

def corr(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    aa = a - a.mean(); bb = b - b.mean()
    return float((aa*bb).sum() / (np.linalg.norm(aa)*np.linalg.norm(bb) + 1e-9))

print("=== native (fp32-residual) vs HF (bf16-residual) per-layer at step 15 ===")
prev = None
for l in range(16):
    nh = ncomp[l, 6][:d].astype(np.float32)
    b = hs[l + 1].numpy()
    c = corr(nh, b)
    md = float(np.abs(nh - b).max())
    # md growth vs upstream: report md and the delta from the previous layer
    dmd = f"  dmd={md - prev:.5f}" if prev is not None else ""
    print(f"  L{l:2d}: corr={c:.7f} md={md:.5f}{dmd}")
    prev = md
