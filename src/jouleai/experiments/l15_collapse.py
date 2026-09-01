"""Pin down the native L15 FFN collapse at decode step 0 (q4).
Compares native L15 ffn_in/router/ffn_out against HF, and computes what the
FFN SHOULD produce given the native router weights (to isolate whether the
collapse is routing-weight or down-proj/orchestration).
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
tok0 = 510

# ---- HF L15 internals at step 0 ----
with torch.no_grad():
    pref = torch.cat([ids["input_ids"], torch.tensor([[tok0]])], dim=1)
    outs = hf.model(input_ids=pref, use_cache=True, output_hidden_states=True)
    l15 = hf.model.layers[15]
    hf_h15 = outs.hidden_states[15][0, -1].float()
    from transformers.models.olmoe.modeling_olmoe import apply_rotary_pos_emb
    xb = l15.input_layernorm(outs.hidden_states[15])[0, -1].float().unsqueeze(0).unsqueeze(0).to(hf.dtype)
    qq = l15.self_attn.q_norm(l15.self_attn.q_proj(xb))
    kk = l15.self_attn.k_norm(l15.self_attn.k_proj(xb))
    vv = l15.self_attn.v_proj(xb)
    pos_id = torch.tensor([[25]])
    pe = hf.model.rotary_emb(outs.hidden_states[15], pos_id)
    qq = qq.view(1, 1, -1, 128).transpose(1, 2)
    kk = kk.view(1, 1, -1, 128).transpose(1, 2)
    vv = vv.view(1, 1, -1, 128).transpose(1, 2)
    qq, kk = apply_rotary_pos_emb(qq, kk, pe[0].to(qq.dtype), pe[1].to(qq.dtype))
    hf_k = outs.past_key_values.layers[15].keys[:, :, :26].float()
    hf_v = outs.past_key_values.layers[15].values[:, :, :26].float()
    o = torch.nn.functional.scaled_dot_product_attention(qq.float(), hf_k, hf_v, is_causal=False)
    o = o.transpose(1, 2).reshape(1, 1, -1)
    hf_attn15 = (o @ l15.self_attn.o_proj.weight.T.float())[0, 0].float()
    hf_h_attn15 = hf_h15 + hf_attn15
    hf_ffnin15 = l15.post_attention_layernorm(hf_h_attn15.unsqueeze(0).unsqueeze(0))[0, 0].float()
    moe15 = l15.mlp(hf_ffnin15.unsqueeze(0).unsqueeze(0).to(hf.dtype))
    hf_ffn15 = (moe15[0] if isinstance(moe15, tuple) else moe15)[0, 0].float()
    gl = hf_ffnin15.unsqueeze(0).to(hf.dtype) @ l15.mlp.gate.weight.T
    probs_hf = torch.softmax(gl.float(), -1)[0]
    top_hf = torch.topk(probs_hf, 8)
    print(f"HF L15 ffn_in norm={hf_ffnin15.norm():.3f} ffn_out norm={hf_ffn15.norm():.3f}")
    print(f"HF L15 top8={top_hf.indices.tolist()} w={[round(float(x),4) for x in top_hf.values.tolist()]}")

# ---- native L15 internals ----
from jouleai.native.decoder3 import NativeDecoder
nd = NativeDecoder(MD)
nd.reset()
nd.prefill(ids["input_ids"][0].tolist())
ncomp = nd.debug_decode_layers(P, tok0)
d = nd.cfg.d
nat_ffnin15 = ncomp[15, 3][:d].astype(np.float32)
nat_router15 = ncomp[15, 4][:64].astype(np.float32)
nat_ffnout15 = ncomp[15, 5][:d].astype(np.float32)
nat_h15 = ncomp[15, 2][:d].astype(np.float32)
print(f"\nnative L15 ffn_in norm={np.linalg.norm(nat_ffnin15):.3f} "
      f"(HF {hf_ffnin15.norm():.3f})")
print(f"native L15 ffn_in vs HF corr={np.corrcoef(nat_ffnin15, hf_ffnin15.numpy())[0,1]:.5f}")
print(f"native L15 router vs HF corr={np.corrcoef(nat_router15, gl[0].float().numpy())[0,1]:.5f}")
print(f"native L15 ffn_out norm={np.linalg.norm(nat_ffnout15):.4f} "
      f"(HF {hf_ffn15.norm():.4f})")
print(f"native L15 h_after_attn norm={np.linalg.norm(nat_h15):.3f} "
      f"(HF {hf_h_attn15.norm():.3f})")

# native router top-8
probs_nat = np.exp(nat_router15 - nat_router15.max())
probs_nat /= probs_nat.sum()
top_nat = np.argsort(probs_nat)[::-1][:8]
print(f"native L15 top8={top_nat.tolist()} "
      f"w={[round(float(probs_nat[i]),4) for i in top_nat.tolist()]}")
# native weights match HF?
print(f"shared top8: {sorted(set(top_nat.tolist()) & set(top_hf.indices.tolist()))}")

# What SHOULD native ffn_out be given native router weights?
# Compute per-expert outputs with the native ffn_in and the Q4 experts, then
# combine with the native top-8 weights; compare to the native ffn_out norm.
from jouleai.storage.q4_store import Q4ExpertPool
pool = Q4ExpertPool(MD, nd.cfg.L, nd.cfg.E, budget_bytes=8 << 30, raw=True)
from jouleai.native import kernel
acc = np.zeros(d, np.float32)
for i, e in enumerate(top_nat.tolist()):
    sc_g, pk_g, n_g = pool.gather(15, int(e))[0]
    sc_u, pk_u, n_u = pool.gather(15, int(e))[1]
    sc_d, pk_d, n_d = pool.gather(15, int(e))[2]
    G = torch.empty(1, n_g // d, dtype=torch.float32)
    kernel.q4_gemm(G, torch.from_numpy(nat_ffnin15).unsqueeze(0), pk_g, sc_g, n_g // d, d)
    U = torch.empty(1, n_u // d, dtype=torch.float32)
    kernel.q4_gemm(U, torch.from_numpy(nat_ffnin15).unsqueeze(0), pk_u, sc_u, n_u // d, d)
    act = torch.nn.functional.silu(G) * U
    D = torch.empty(1, d, dtype=torch.float32)
    kernel.q4_gemm(D, act, pk_d, sc_d, d, nd.cfg.intermediate)
    acc += probs_nat[i] * D[0].numpy()
print(f"\nmanual q4 ffn_out (native router w) norm={np.linalg.norm(acc):.4f}")
print(f"native kernel ffn_out norm={np.linalg.norm(nat_ffnout15):.4f}")
print(f"corr(manual, native kernel ffn_out)={np.corrcoef(acc, nat_ffnout15)[0,1]:.5f}")
