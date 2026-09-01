"""Component-level bisect at the drift divergence position (OLMoE).

Runs HF step-by-step to the divergence step k (consuming HF's own tokens),
capturing per-layer components via module hooks at that step; then runs the
native kernel's debug_decode_layers at the same absolute position (replaying
the identical HF token) and compares component-by-component per layer.

The layer where a component's corr drops below ~0.99 while its upstream
inputs still match is the drift source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.native.decoder3 import NativeDecoder  # noqa: E402

_QUERIES = [
    "What is the capital of France? Answer in one sentence.",
    "Explain photosynthesis in simple terms.",
]

NAMES = ("attn_in", "attn_out", "h_attn", "ffn_in", "router", "ffn_out",
         "h_ffn")


def main() -> int:
    model_dir = Path(sys.argv[1])
    qi = 0
    if "--query" in sys.argv:
        qi = int(sys.argv[sys.argv.index("--query") + 1])
    q = _QUERIES[qi]
    k = 7
    if "--step" in sys.argv:
        k = int(sys.argv[sys.argv.index("--step") + 1])

    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    tok = AutoTokenizer.from_pretrained(model_dir)
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    hf.eval()
    t = tok.apply_chat_template([{"role": "user", "content": q}],
                                add_generation_prompt=True, tokenize=False)
    ids = tok(t, return_tensors="pt")
    P = ids["input_ids"].shape[1]
    N = k + 2
    hf_tokens: list[int] = []
    with torch.no_grad():
        outs = hf.model(input_ids=ids["input_ids"], use_cache=True)
        past = outs.past_key_values
        lg = hf.lm_head(hf.model.norm(outs.last_hidden_state))[:, -1]
        hf_tokens.append(int(lg.argmax()))
        for _ in range(N - 1):
            cur = torch.tensor([[hf_tokens[-1]]])
            outs = hf.model(input_ids=cur, past_key_values=past, use_cache=True)
            past = outs.past_key_values
            lg = hf.lm_head(hf.model.norm(outs.last_hidden_state))[:, -1]
            hf_tokens.append(int(lg.argmax()))
    print(f"[hf] prompt_len={P} tokens[:{N}] = {hf_tokens}")

    L = len(hf.model.layers)
    d = hf.config.hidden_size
    E = hf.config.num_experts

    # ---- prefix KV (positions < P+k-1) for the step-k forward ----
    dc = DynamicCache()
    for l in range(L):
        kk = past.layers[l].keys[:, :, :P + k - 1].clone()
        vv = past.layers[l].values[:, :, :P + k - 1].clone()
        dc.update(kk, vv, l)

    # ---- hooks capture components at step k ----
    def _arg0(args, kwargs, key="hidden_states"):
        if args:
            return args[0]
        return kwargs[key]

    caps: dict[int, dict] = {}
    for l in range(L):
        caps[l] = {}
        sa = hf.model.layers[l].self_attn
        mlp = hf.model.layers[l].mlp
        sa.register_forward_pre_hook(
            lambda mod, args, kwargs, l=l: caps[l].__setitem__(
                "attn_in", _arg0(args, kwargs)[0, 0].float().clone()),
            with_kwargs=True)
        sa.register_forward_hook(
            lambda mod, args, kwargs, out, l=l: caps[l].__setitem__(
                "attn_out", out[0][0, 0].float().clone()),
            with_kwargs=True)
        mlp.register_forward_pre_hook(
            lambda mod, args, kwargs, l=l: caps[l].__setitem__(
                "ffn_in", _arg0(args, kwargs)[0, 0].float().clone()),
            with_kwargs=True)
        mlp.register_forward_hook(
            lambda mod, args, kwargs, out, l=l: caps[l].__setitem__(
                "ffn_out", out[0, 0].float().clone()),
            with_kwargs=True)
        gate = mlp.gate
        gate.register_forward_hook(
            lambda mod, args, kwargs, out, l=l: caps[l].__setitem__(
                "router", out[0].float().clone()),
            with_kwargs=True)

    if k == 0:
        h0 = hf.model.embed_tokens(ids["input_ids"])[:, -1:]
    else:
        h0 = hf.model.embed_tokens(torch.tensor([[hf_tokens[k - 1]]]))
    pos_id = torch.tensor([[P + k - 1]])
    with torch.no_grad():
        pe = hf.model.rotary_emb(h0, pos_id)
        pe = (pe[0].to(h0.dtype), pe[1].to(h0.dtype))
        h = h0
        for l in range(L):
            h = hf.model.layers[l](h, position_ids=pos_id,
                                   past_key_values=dc, use_cache=True,
                                   position_embeddings=pe)

    # h_attn / h_ffn from a second pass with layer-level hooks
    caps2: dict[int, dict] = {}
    for l in range(L):
        caps2[l] = {}
        lo = hf.model.layers[l]
        lo.register_forward_pre_hook(
            lambda mod, args, kwargs, l=l: caps2[l].__setitem__(
                "h_in", _arg0(args, kwargs)[0, 0].float().clone()),
            with_kwargs=True)
        lo.register_forward_hook(
            lambda mod, args, kwargs, out, l=l: caps2[l].__setitem__(
                "h_out", out[0, 0].float().clone()),
            with_kwargs=True)
    with torch.no_grad():
        h = h0
        for l in range(L):
            h = hf.model.layers[l](h, position_ids=torch.tensor([[P + k - 1]]),
                        past_key_values=dc, use_cache=True,
                        position_embeddings=pe)
    for l in range(L):
        caps[l]["h_attn"] = (caps2[l]["h_in"].float()
                             + caps[l]["attn_out"].float())
        caps[l]["h_ffn"] = (caps2[l]["h_out"].float())

    # ---- native replay ----
    nd = NativeDecoder(model_dir)
    nd.reset()
    nd.prefill(ids["input_ids"][0].tolist())
    for j in range(k):
        nd.decode_token(hf_tokens[j])
    ncomp = nd.debug_decode_layers(P + k - 1, hf_tokens[k - 1])
    rlen = ncomp.shape[2]

    def cmp(name, hf_v, nat_v, n):
        a = hf_v[:n].float().numpy()
        b = nat_v[:n].astype(np.float32)
        corr = np.corrcoef(a, b)[0, 1]
        md = np.abs(a - b).max()
        return corr, md

    print(f"\n=== per-layer component comparison at decode position "
          f"{P + k - 1} (step {k}) ===")
    print(f"  {'L':>2} | " + " | ".join(f"{nm:>14}" for nm in NAMES))
    for l in range(L):
        row = []
        for nm in NAMES:
            n = E if nm == "router" else d
            corr, md = cmp(nm, caps[l][nm], ncomp[l, NAMES.index(nm)], n)
            row.append(f"{corr:6.4f}/{md:7.3f}")
        print(f"  {l:2d} | " + " | ".join(row))

    print("\n=== ffn_in -> router -> ffn_out corr drop ===")
    for l in range(L):
        ci = cmp("ffn_in", caps[l]["ffn_in"], ncomp[l, 3], d)[0]
        cr = cmp("router", caps[l]["router"], ncomp[l, 4], E)[0]
        co = cmp("ffn_out", caps[l]["ffn_out"], ncomp[l, 5], d)[0]
        flag = " <<< DROP" if (co < ci - 0.02 and ci > 0.999) else ""
        print(f"  L{l:2d}: ffn_in={ci:6.4f} router={cr:6.4f} "
              f"ffn_out={co:6.4f}{flag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
