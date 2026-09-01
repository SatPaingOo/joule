"""Focused: with bf16-exact experts, where does the step-15 state diverge?

The bf16 tier removes quantization, so any native-vs-HF difference at step 15
is a STATE bug (KV / routing / attention), not noise. Compare per-layer:
  - decode-step KV at positions 38-40 (the divergence region)
  - L15 router logits/topk at step 15
  - per-layer residuals at step 15
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

# HF 16-step greedy + full KV + hidden states
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
print("HF 16 tokens:", hf_tokens)

# HF reference hidden at step 15 (position P+14): prefix = prompt + first 15
with torch.no_grad():
    pref = torch.cat([ids["input_ids"], torch.tensor([hf_tokens[:15]])], dim=1)
    outs = hf.model(input_ids=pref, use_cache=True, output_hidden_states=True)
    hs = [x[0, -1].float() for x in outs.hidden_states]
    l15 = hf.model.layers[15]
    hf_ffnin15 = l15.post_attention_layernorm(outs.hidden_states[16])[0, -1].float()
    gl = hf_ffnin15.unsqueeze(0).to(hf.dtype) @ l15.mlp.gate.weight.T
    probs_hf = torch.softmax(gl.float(), -1)[0]
    top_hf = torch.topk(probs_hf, 8)

from jouleai.native.decoder3 import NativeDecoder
nd = NativeDecoder(MD, precision="bf16")
nd.reset()
nd.prefill(ids["input_ids"][0].tolist())
for j in range(15):
    nd.decode_token(hf_tokens[j])

# 1. decode-step KV comparison at positions 37-39 (bf16 experts -> should be exact if state right)
print("\n=== decode-step KV maxdiff vs HF (bf16 experts) ===")
for pos in range(P + 12, P + 15):
    row = []
    for l in [0, 7, 15]:
        nk = nd._seq_kv["seq0"][l][0][pos]
        hk = past.layers[l].keys[0, :, pos].float().numpy()
        row.append(f"L{l}:{np.abs(nk-hk).max():.5f}")
    print(f"pos {pos}: " + "  ".join(row))

# 2. native L15 router at step 15
ncomp = nd.debug_decode_layers(P + 14, hf_tokens[14])
d = nd.cfg.d
nat_ffnin15 = ncomp[15, 3][:d].astype(np.float32)
nat_router15 = ncomp[15, 4][:64].astype(np.float32)
print("\n=== L15 at step 15 (divergence point) ===")
print(f"ffn_in corr: {np.corrcoef(nat_ffnin15, hf_ffnin15.numpy())[0,1]:.6f} "
      f"md: {np.abs(nat_ffnin15-hf_ffnin15.numpy()).max():.4f}")
print(f"router corr: {np.corrcoef(nat_router15, gl[0].float().numpy())[0,1]:.6f}")
probs_nat = np.exp(nat_router15 - nat_router15.max()); probs_nat /= probs_nat.sum()
top_nat = np.argsort(probs_nat)[::-1][:8]
print("HF  top8:", top_hf.indices.tolist())
print("nat top8:", top_nat.tolist())
print("HF  w:", [round(float(x),4) for x in top_hf.values.tolist()])
print("nat w:", [round(float(probs_nat[i]),4) for i in top_nat.tolist()])

# 3. per-layer residuals at step 15
print("\n=== per-layer residual vs HF at step 15 (bf16) ===")
for l in range(16):
    nh = ncomp[l, 6][:d].astype(np.float32)
    b = hs[l + 1].numpy()
    aa = nh - nh.mean(); bb = b - b.mean()
    corr = float((aa*bb).sum() / (np.linalg.norm(aa)*np.linalg.norm(bb) + 1e-9))
    flag = " <<<" if corr < 0.9995 else ""
    print(f"  L{l:2d}: corr={corr:.6f} md={np.abs(nh-b).max():.5f}{flag}")
