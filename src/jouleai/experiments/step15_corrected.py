"""Corrected step-15 analysis: hook-captured HF references (no off-by-one).

HF ffn_in at layer l = post_attention_layernorm(hidden_states[l] + attn_out_l)
— NOT post_attention_layernorm(hidden_states[l+1]) which is the post-FFN
residual. Hooks capture the true pre-mlp input/output.
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

# HF 16 tokens
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

# HF at step 15: prefix = prompt + first 15 tokens; hooks capture per-layer
# ffn_in (pre-mlp) + ffn_out (post-mlp) + attn_out
caps = {}
pref = torch.cat([ids["input_ids"], torch.tensor([hf_tokens[:15]])], dim=1)
for l in range(16):
    caps[l] = {}
    mlp = hf.model.layers[l].mlp
    sa = hf.model.layers[l].self_attn
    def pre_mlp(mod, args, kwargs, l=l):
        x = args[0] if args else kwargs["hidden_states"]
        caps[l]["ffn_in"] = x[0, -1].float().clone()
    def post_mlp(mod, args, kwargs, out, l=l):
        o = out[0] if isinstance(out, tuple) else out
        caps[l]["ffn_out"] = o[0, -1].float().clone()
    def post_attn(mod, args, kwargs, out, l=l):
        caps[l]["attn_out"] = out[0][0, -1].float().clone()
    mlp.register_forward_pre_hook(pre_mlp, with_kwargs=True)
    mlp.register_forward_hook(post_mlp, with_kwargs=True)
    sa.register_forward_hook(post_attn, with_kwargs=True)
with torch.no_grad():
    outs = hf.model(input_ids=pref, use_cache=True, output_hidden_states=True)
    hs = [x[0, -1].float() for x in outs.hidden_states]
    # HF router logits per layer (from the true ffn_in)
    router_hf = {}
    for l in range(16):
        xb = caps[l]["ffn_in"].unsqueeze(0).to(hf.dtype)
        router_hf[l] = (xb @ hf.model.layers[l].mlp.gate.weight.T).float()[0]

# Native bf16: replay 15 steps, capture per-layer ffn_in/router/ffn_out
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

print("=== corrected per-layer at step 15 (bf16): ffn_in / router / ffn_out ===")
print(f"  {'L':>2} | {'ffn_in c/md':>15} | {'router c':>9} | {'ffn_out c/md':>16} | {'resid c':>8}")
for l in range(16):
    nf_in = ncomp[l, 3][:d].astype(np.float32)
    n_rt = ncomp[l, 4][:64].astype(np.float32)
    nf_out = ncomp[l, 5][:d].astype(np.float32)
    nres = ncomp[l, 6][:d].astype(np.float32)
    c_in = corr(nf_in, caps[l]["ffn_in"].numpy())
    m_in = float(np.abs(nf_in - caps[l]["ffn_in"].numpy()).max())
    c_rt = corr(n_rt, router_hf[l].numpy())
    c_out = corr(nf_out, caps[l]["ffn_out"].numpy())
    m_out = float(np.abs(nf_out - caps[l]["ffn_out"].numpy()).max())
    c_res = corr(nres, hs[l+1].numpy())
    flag = " <<<" if c_in < 0.999 or c_rt < 0.99 else ""
    print(f"  {l:2d} | {c_in:8.5f}/{m_in:7.4f} | {c_rt:9.5f} | "
          f"{c_out:8.5f}/{m_out:7.4f} | {c_res:8.5f}{flag}")
