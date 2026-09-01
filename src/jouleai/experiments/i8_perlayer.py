"""i8 native per-layer residual vs HF at step 0 — find where i8 breaks."""
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
    hs = [x[0, -1].float() for x in outs.hidden_states]

from jouleai.native.decoder3 import NativeDecoder
for prec in ("q4", "i8"):
    nd = NativeDecoder(MD, precision=prec)
    nd.reset()
    nd.prefill(ids["input_ids"][0].tolist())
    ncomp = nd.debug_decode_layers(P, tok0)
    d = nd.cfg.d
    print(f"=== {prec}: native per-layer residual vs HF (step 0) ===")
    for l in range(16):
        nh = ncomp[l, 6][:d].astype(np.float32)
        b = hs[l + 1].numpy()
        corr = float(np.corrcoef(nh, b)[0, 1])
        md = float(np.abs(nh - b).max())
        flag = " <<<" if corr < 0.995 else ""
        print(f"  L{l:2d}: corr={corr:.5f} md={md:.4f} "
              f"norm={np.linalg.norm(nh):6.2f}/{np.linalg.norm(b):6.2f}{flag}")
    del nd
    gc.collect()
