"""True L15 ffn_out via HF module hooks at decode step 0, vs native q4/i8."""
import sys
sys.path.insert(0, "src")
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
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
L = len(hf.model.layers)

# ---- HF: prefix = prompt + [510], run the REAL model, capture L15 ffn_out ----
caps = {}
with torch.no_grad():
    pref = torch.cat([ids["input_ids"], torch.tensor([[tok0]])], dim=1)
    outs = hf.model(input_ids=pref, use_cache=True, output_hidden_states=True)
    hs = [x[0, -1].float() for x in outs.hidden_states]
    l15 = hf.model.layers[15]
    caps["ffn_in"] = None
    caps["ffn_out"] = None
    def pre(mod, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        caps["ffn_in"] = x[0, -1].float().clone()
    def post(mod, args, kwargs, out):
        o = out[0] if isinstance(out, tuple) else out
        caps["ffn_out"] = o[0, -1].float().clone()
    l15.mlp.register_forward_pre_hook(pre, with_kwargs=True)
    l15.mlp.register_forward_hook(post, with_kwargs=True)
    # rerun the model forward to capture L15 mlp in/out (idempotent KV)
    outs2 = hf.model(input_ids=pref, use_cache=True, output_hidden_states=True)
    hf_ffnin15 = caps["ffn_in"]
    hf_ffn15 = caps["ffn_out"]
print(f"HF L15 ffn_in norm={hf_ffnin15.norm():.3f} ffn_out norm={hf_ffn15.norm():.3f}")
print(f"HF L15 residual (h16) norm={hs[16].norm():.3f}")

# ---- native q4/i8 L15 ----
from jouleai.native.decoder3 import NativeDecoder
for prec in ("q4", "i8"):
    nd = NativeDecoder(MD, precision=prec)
    nd.reset()
    nd.prefill(ids["input_ids"][0].tolist())
    ncomp = nd.debug_decode_layers(P, tok0)
    d = nd.cfg.d
    nat_ffnin15 = ncomp[15, 3][:d].astype(np.float32)
    nat_ffnout15 = ncomp[15, 5][:d].astype(np.float32)
    c_in = float(np.corrcoef(nat_ffnin15, hf_ffnin15.numpy())[0, 1])
    c_out = float(np.corrcoef(nat_ffnout15, hf_ffn15.numpy())[0, 1])
    print(f"{prec} L15 ffn_in corr={c_in:.5f} md={np.abs(nat_ffnin15-hf_ffnin15.numpy()).max():.3f}")
    print(f"{prec} L15 ffn_out corr={c_out:.5f} md={np.abs(nat_ffnout15-hf_ffn15.numpy()).max():.4f} "
          f"norm={np.linalg.norm(nat_ffnout15):.3f}/{hf_ffn15.norm():.3f}")
    del nd
    gc.collect()
