"""Compare native q4 vs i8 L0 ffn_out against HF at decode step 0."""
import sys
sys.path.insert(0, "src")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
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
tok0 = 510

with torch.no_grad():
    pref = torch.cat([ids["input_ids"], torch.tensor([[tok0]])], dim=1)
    outs = hf.model(input_ids=pref, use_cache=True, output_hidden_states=True)
    l0 = hf.model.layers[0]
    hf_attn_in = l0.input_layernorm(outs.hidden_states[0])[0, -1].float()
    xb = hf_attn_in.unsqueeze(0).unsqueeze(0).to(hf.dtype)
    from transformers.models.olmoe.modeling_olmoe import apply_rotary_pos_emb
    qq = l0.self_attn.q_norm(l0.self_attn.q_proj(xb))
    kk = l0.self_attn.k_norm(l0.self_attn.k_proj(xb))
    vv = l0.self_attn.v_proj(xb)
    pos_id = torch.tensor([[25]])
    pe = hf.model.rotary_emb(outs.hidden_states[0], pos_id)
    qq = qq.view(1, 1, -1, 128).transpose(1, 2)
    kk = kk.view(1, 1, -1, 128).transpose(1, 2)
    vv = vv.view(1, 1, -1, 128).transpose(1, 2)
    qq, kk = apply_rotary_pos_emb(qq, kk, pe[0].to(qq.dtype), pe[1].to(qq.dtype))
    hf_k = outs.past_key_values.layers[0].keys[:, :, :26].float()
    hf_v = outs.past_key_values.layers[0].values[:, :, :26].float()
    o = torch.nn.functional.scaled_dot_product_attention(
        qq.float(), hf_k, hf_v, is_causal=False)
    o = o.transpose(1, 2).reshape(1, 1, -1)
    hf_attn_out = (o @ l0.self_attn.o_proj.weight.T.float())[0, 0].float()
    hf_h1 = outs.hidden_states[0][0, -1].float() + hf_attn_out
    hf_ffnin = l0.post_attention_layernorm(
        hf_h1.unsqueeze(0).unsqueeze(0))[0, 0].float()
    moe_out = l0.mlp(hf_ffnin.unsqueeze(0).unsqueeze(0).to(hf.dtype))
    hf_ffn_out = (moe_out[0] if isinstance(moe_out, tuple) else moe_out)[0, 0].float()

from jouleai.native.decoder3 import NativeDecoder
for prec in ("q4", "i8"):
    nd = NativeDecoder(MD, precision=prec)
    nd.reset()
    nd.prefill(ids["input_ids"][0].tolist())
    ncomp = nd.debug_decode_layers(P, tok0)
    d = nd.cfg.d
    nat_ffnout = ncomp[0, 5][:d].astype(np.float32)
    corr = float(np.corrcoef(nat_ffnout, hf_ffn_out.numpy())[0, 1])
    md = float(np.abs(nat_ffnout - hf_ffn_out.numpy()).max())
    print(f"{prec} L0 ffn_out vs HF: corr={corr:.6f} md={md:.5f} "
          f"norm={np.linalg.norm(nat_ffnout):.5f}/{np.linalg.norm(hf_ffn_out.numpy()):.5f}")
    del nd
    gc.collect()
