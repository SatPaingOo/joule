"""KV-aware skip: skip the FFN (experts) of chosen layers but KEEP attention+KV.

For MoE the FFN is the bandwidth bottleneck (experts = 90% of weight reads),
so skipping ONLY the FFN of low-influence layers keeps KV intact while cutting
the expert reads. Tests quality vs full for FFN-skip on the first k layers.
"""
import sys
sys.path.insert(0, "src")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

MD = "models/OLMoE-1B-7B-0824-Instruct"
tk = AutoTokenizer.from_pretrained(MD)
hf = AutoModelForCausalLM.from_pretrained(MD, dtype=torch.bfloat16)
hf.eval()
q = "What is the capital of France? Answer in one sentence."
t = tk.apply_chat_template([{"role": "user", "content": q}],
                           add_generation_prompt=True, tokenize=False)
ids = tk(t, return_tensors="pt")

hooks = []


def install_ffn_skip(k: int):
    """Skip (identity) the MLP of the first k layers; attention+KV still run."""
    global hooks
    for h in hooks:
        h.remove()
    hooks = []
    for l in range(k):
        mlp = hf.model.layers[l].mlp
        hooks.append(mlp.register_forward_hook(
            lambda mod, args, kwargs, out, l=l: args[0] if args else
            kwargs["hidden_states"], with_kwargs=True))


def run(k: int, n_tokens: int = 64):
    install_ffn_skip(k)
    cache = None
    out = []
    cur = ids["input_ids"]
    t0 = time.perf_counter()
    with torch.no_grad():
        for step in range(n_tokens):
            outs = hf.model(input_ids=cur, use_cache=True,
                            past_key_values=cache)
            cache = outs.past_key_values
            lg = hf.lm_head(hf.model.norm(outs.last_hidden_state))[:, -1]
            nxt = int(lg.argmax())
            out.append(nxt)
            cur = torch.tensor([[nxt]])
    wall = time.perf_counter() - t0
    text = tk.decode(out, skip_special_tokens=True)
    reps = 0
    seen = set()
    for i in range(len(out) - 4):
        gram = tuple(out[i:i + 4])
        if gram in seen:
            reps += 1
        seen.add(gram)
    return out, text, reps, wall


for k in (0, 2, 4, 6):
    out, text, reps, wall = run(k)
    print(f"=== FFN-skip first-{k} layers ({64/wall:.1f} tok/s) reps={reps} ===")
    print(f"  {text[:110]!r}")
