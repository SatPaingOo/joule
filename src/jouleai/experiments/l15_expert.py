"""Compare native L15 per-expert down output vs HF for the top experts."""
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

# HF L15 ffn_in + true ffn_out
with torch.no_grad():
    pref = torch.cat([ids["input_ids"], torch.tensor([[510]])], dim=1)
    outs = hf.model(input_ids=pref, use_cache=True, output_hidden_states=True)
    hs = [x[0, -1].float() for x in outs.hidden_states]
    l15 = hf.model.layers[15]
    from transformers.models.olmoe.modeling_olmoe import apply_rotary_pos_emb
    xb = l15.input_layernorm(outs.hidden_states[15])[0, -1].float().unsqueeze(0).unsqueeze(0).to(hf.dtype)
    qq = l15.self_attn.q_norm(l15.self_attn.q_proj(xb))
    kk = l15.self_attn.k_norm(l15.self_attn.k_proj(xb))
    vv = l15.self_attn.v_proj(xb)
    pos_id = torch.tensor([[25]])
    pe = hf.model.rotary_emb(outs.hidden_states[15], pos_id)
    qq = qq.view(1,1,-1,128).transpose(1,2); kk = kk.view(1,1,-1,128).transpose(1,2); vv = vv.view(1,1,-1,128).transpose(1,2)
    qq, kk = apply_rotary_pos_emb(qq, kk, pe[0].to(qq.dtype), pe[1].to(qq.dtype))
    hf_k = outs.past_key_values.layers[15].keys[:, :, :26].float()
    hf_v = outs.past_key_values.layers[15].values[:, :, :26].float()
    o = torch.nn.functional.scaled_dot_product_attention(qq.float(), hf_k, hf_v, is_causal=False)
    o = o.transpose(1,2).reshape(1,1,-1)
    hf_attn15 = (o @ l15.self_attn.o_proj.weight.T.float())[0,0].float()
    hf_h_attn = hs[15] + hf_attn15
    hf_ffnin15 = l15.post_attention_layernorm(hf_h_attn.unsqueeze(0).unsqueeze(0))[0,0].float()
    # HF routing
    gl = hf_ffnin15.unsqueeze(0).to(hf.dtype) @ l15.mlp.gate.weight.T
    probs_hf = torch.softmax(gl.float(), -1)[0]
    top_hf = torch.topk(probs_hf, 8)
    print("HF L15 top8:", top_hf.indices.tolist())
    print("HF L15 w:", [round(float(x),4) for x in top_hf.values.tolist()])
    # HF per-expert outputs (expert e -> down contribution)
    for e in top_hf.indices[:3].tolist():
        ex = l15.mlp.experts
        # gate_up fused
        gu = ex.gate_up_proj[int(e)]
        dn = ex.down_proj[int(e)]
        g, u = (hf_ffnin15.unsqueeze(0).unsqueeze(0).to(hf.dtype) @ gu.T).chunk(2, dim=-1)
        act = torch.nn.functional.silu(g) * u
        y = (act @ dn.T)[0, 0].float()
        print(f"  HF expert {e}: out norm {y.norm():.3f}")

# Native L15 ffn_in + per-expert via the kernel's q4 row dots
from jouleai.native.decoder3 import NativeDecoder
nd = NativeDecoder(MD)
nd.reset()
nd.prefill(ids["input_ids"][0].tolist())
ncomp = nd.debug_decode_layers(P, 510)
d = nd.cfg.d
nat_ffnin15 = ncomp[15, 3][:d].astype(np.float32)
print(f"\nnative L15 ffn_in norm: {np.linalg.norm(nat_ffnin15):.3f} "
      f"(HF {hf_ffnin15.norm():.3f})")
print("native L15 ffn_in vs HF corr:",
      float(np.corrcoef(nat_ffnin15, hf_ffnin15.numpy())[0,1]))
# native top-8
probs_nat = np.exp(nat_ffnin15 @ nd.gate_w[15].astype(np.float32).T)
probs_nat /= probs_nat.sum()
top_nat = np.argsort(probs_nat)[::-1][:8]
print("native L15 top8:", top_nat.tolist())
print("native L15 w:", [round(float(probs_nat[i]),4) for i in top_nat.tolist()])

# Native per-expert down via the raw Q4 store + kernel q4_gemm
from jouleai.storage.q4_store import Q4ExpertPool
from jouleai.native import kernel
pool = Q4ExpertPool(MD, nd.cfg.L, nd.cfg.E, budget_bytes=8 << 30, raw=True)
for e in top_hf.indices[:3].tolist():
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
    print(f"  native expert {e} (q4_gemm): out norm {D[0].norm():.3f}")
