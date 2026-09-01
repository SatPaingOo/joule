"""Verify: does bf16-rounding the FFN intermediates match HF exactly?

HF expert FFN (all bf16): gate,up = linear(x) -> chunk; act = silu(gate)*up;
y = linear(act, down); out += weight * y. Every output rounds to bf16.
The native bf16 tier keeps fp32. Test: take the native L5 ffn_in at step 15,
compute the expert FFN with (a) fp32 (native-style) vs (b) bf16-rounded
(HF-style), compare both to HF's hook-captured L5 ffn_out.
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

# HF hook: L5 ffn_in + ffn_out + routing at step 15
caps = {}
l5 = hf.model.layers[5]
def pre(mod, args, kwargs):
    x = args[0] if args else kwargs["hidden_states"]
    caps["ffn_in"] = x[0, -1].float().clone()
def post(mod, args, kwargs, out):
    o = out[0] if isinstance(out, tuple) else out
    caps["ffn_out"] = o[0, -1].float().clone()
l5.mlp.register_forward_pre_hook(pre, with_kwargs=True)
l5.mlp.register_forward_hook(post, with_kwargs=True)
pref = torch.cat([ids["input_ids"], torch.tensor([hf_tokens[:15]])], dim=1)
with torch.no_grad():
    outs = hf.model(input_ids=pref, use_cache=True)
hf_ffn_in = caps["ffn_in"]
hf_ffn_out = caps["ffn_out"]
print("HF L5 ffn_in norm:", hf_ffn_in.norm().item(), "ffn_out norm:", hf_ffn_out.norm().item())

# HF routing at L5 (from the true ffn_in)
with torch.no_grad():
    gl = hf_ffn_in.unsqueeze(0).to(hf.dtype) @ l5.mlp.gate.weight.T
    probs = torch.softmax(gl.float(), -1)[0]
    top = torch.topk(probs, 8)
    idx_hf = top.indices.tolist()
    w_hf = top.values.float()
print("HF L5 top8:", idx_hf)

# native bf16 L5 ffn_in at step 15
from jouleai.native.decoder3 import NativeDecoder
nd = NativeDecoder(MD, precision="bf16")
nd.reset()
nd.prefill(ids["input_ids"][0].tolist())
for j in range(15):
    nd.decode_token(hf_tokens[j])
ncomp = nd.debug_decode_layers(P + 14, hf_tokens[14])
d = nd.cfg.d
nat_ffn_in = torch.from_numpy(ncomp[5, 3][:d].astype(np.float32))
print("nat L5 ffn_in vs HF corr:", float(torch.corrcoef(torch.stack([nat_ffn_in, hf_ffn_in]))[0,1]))

# native routing (use HF's index/weights to isolate the FFN math)
# compute the expert FFN both ways on the native ffn_in with HF's routing
def expert_ffn_fp32(x_fp32, e):
    gu = l5.mlp.experts.gate_up_proj[e]
    dn = l5.mlp.experts.down_proj[e]
    g, u = (x_fp32.unsqueeze(0) @ gu.T.float()).chunk(2, dim=-1)
    act = torch.nn.functional.silu(g) * u
    return (act @ dn.T.float())[0]

def expert_ffn_bf16(x_bf16, e):
    gu = l5.mlp.experts.gate_up_proj[e]
    dn = l5.mlp.experts.down_proj[e]
    g, u = (x_bf16.unsqueeze(0) @ gu.T).chunk(2, dim=-1)  # bf16 linear
    act = torch.nn.functional.silu(g) * u                  # bf16
    return (act @ dn.T)[0]                                 # bf16

x_bf16 = nat_ffn_in.to(torch.bfloat16)
out_fp32 = torch.zeros(d)
out_bf16 = torch.zeros(d)
for i, e in enumerate(idx_hf):
    out_fp32 += w_hf[i] * expert_ffn_fp32(nat_ffn_in, e)
    out_bf16 += w_hf[i].to(torch.bfloat16) * expert_ffn_bf16(x_bf16, e)

def corr(a, b):
    aa = a - a.mean(); bb = b - b.mean()
    return float((aa*bb).sum() / (aa.norm()*bb.norm() + 1e-9))
print("\n== L5 ffn_out comparisons (same native input, HF routing) ==")
print(f"fp32-native-style vs HF: corr={corr(out_fp32, hf_ffn_out):.6f} "
      f"md={(out_fp32-hf_ffn_out).abs().max():.5f} norm={out_fp32.norm():.4f}/{hf_ffn_out.norm():.4f}")
print(f"bf16-rounded vs HF:     corr={corr(out_bf16, hf_ffn_out):.6f} "
      f"md={(out_bf16-hf_ffn_out).abs().max():.5f} norm={out_bf16.norm():.4f}/{hf_ffn_out.norm():.4f}")
