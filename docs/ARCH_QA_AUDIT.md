# Joule — Architecture QA Audit (native kernel × registry)
> **Context**: the kernel QA audit (arch-flag drift caught by the verify
> harness). Evidence: [results/VALIDATION_LOG.md](../results/VALIDATION_LOG.md)
> (Entry 67). See [docs/USAGE.md](USAGE.md) (run), [docs/JOULE_PAPER.md](JOULE_PAPER.md) (retrospective).

---

> 2026-08-31 | Scope: "fix kernel for any models arch switch".
> Audited: `arch/registry.py` (spec), `engine/generic_streamer.py` (Python
> forward, HF-verified), `native/decode_kernel.c` + `native/decoder3.py`
> (native full-decode kernel, used by `joule serve --backend native` and the
> `/v1/model/<name>` switch).

## TL;DR

The **Python path** (GenericStreamer) is registry-driven and HF-verified
(qwen2/qwen3/llama/mistral/olmoe/qwen3_moe — Entry 28/29). The **native
kernel** is a *Qwen3-30B-specialized* fast path: it reads `config.json`
directly instead of through the registry, drops the arch flags (QK-norm,
bias, top-k renormalization, llama3 RoPE scaling, router naming), and its
workspace is a fixed-cap static buffer that **silently memory-overflows on
any model outside the 30B's shape envelope** (Mistral-7B and Qwen2.5-7B
already overflow today). Switching models through the native path can
therefore produce wrong tokens or corrupt memory *silently*.

This document lists every gap with severity and the fix that landed.

---

## 1. The flag matrix: registry → Python (verified) → native

| Arch flag | Registry | GenericStreamer (Python) | Native kernel (before) | Severity |
|---|---|---|---|---|
| QK-norm type (none/per_head/whole) | per_head for qwen3/qwen3_moe, whole for olmoe | ✅ implemented | ❌ read raw config (key absent in Qwen3/OLMoE configs → **skipped**) — kernel only had per-head | **HIGH** (wrong attention on every Qwen3/OLMoE native run) |
| QKV bias | qwen2 default true | ✅ | ❌ no bias pointers at all (Qwen2.5 has q/k/v bias) | **HIGH** (wrong attention on Qwen2.5 native) |
| top-k renorm (`norm_topk_prob`) | from config | ✅ | ❌ kernel always renormalizes (only qwen3_moe does in HF; OLMoE/Mixtral must not) | **HIGH** (wrong expert weights on OLMoE/Mixtral native) |
| RoPE scaling (llama3) | `rope_scaling` | ✅ `llama3_inv_freq` | ❌ `_rope_tables` used theta only (Llama-3.2 factor-32 stretch ignored) | **HIGH** (wrong positions on Llama-3.x native) |
| Workspace sizing | — | — | ❌ fixed static caps (d≤8192, H·hd≤8192, m≤12288, E≤512, B≤16); **Mistral-7B (m=14336) and Qwen2.5-7B (m=18944) overflow** | **CRITICAL** (silent memory corruption) |
| Router tensor name | `expert_naming` | `mlp.gate.weight` only | ❌ same hardcode (Mixtral router = `mlp.block_sparse_moe.router.weight`) | MEDIUM (Mixtral native) |
| Expert tensor naming (qwen vs block_sparse_moe) | ✅ | ✅ via q4 store | ✅ via q4 store (conversion-time) | — |
| Sliding-window attention | ❌ not modeled | ❌ causal only (Qwen2.5 has window 32768/131072 — only >window contexts diverge) | ❌ | LOW (latent; short tests unaffected) |
| Prefill lm_head | — | — | ❌ applies lm_head at **every** prefill token (docstring says last-only) | MEDIUM (perf, 1.2 GB/token wasted) |
| Arch capability gate | lists 11 "supported" | ❌ none (gemma/phi/gpt_oss/deepseek would KeyError or route wrong) | ❌ none | MEDIUM (loud-fail needed) |
| `q4_dot_test` debug export | — | — | ❌ hardcoded 128 experts / stride 768 | LOW (debug-only) |

## 2. Registry overclaims (detected ≠ implemented)

`SUPPORTED` advertises gemma / phi / gpt_oss / deepseek, but **no engine
implements them**:

- **deepseek**: MLA attention (latent KV) — the kernel's KV-cache attention
  cannot express it. Worse, `get_spec` maps deepseek → `block_sparse_moe`
  expert naming, but HF DeepSeek uses `mlp.experts.{e}.{gate,up,down}_proj`
  (+ shared expert) — conversion would look up wrong tensor names.
- **phi**: dense MLP is `mlp.fc1/fc2` + gelu (kernel/streamer hardcode
  `gate_proj/up_proj/down_proj` + silu).
- **gemma**: gelu activation + distinct norm style (`gemma_norm` flag is a
  marker nothing consumes).
- **gpt_oss**: not implemented.

Fix: engine-side capability gates (GenericStreamer + NativeDecoder) raise a
clear `ValueError` for these instead of failing silently, and the doc now
separates **detected** from **implemented** families.

## 3. Registry default fragility

`bias_qkv = cfg.get("attention_bias", mt == "qwen2")` defaults **True** for
qwen2 when the key is absent — a qwen2 checkpoint without bias would
`KeyError` on `q_proj.bias`. The decoder now confirms bias presence in the
weight store and falls back to `bias_qkv=False` with a warning.

## 4. Why "coherent answers" hid the native bugs

The native kernel was validated by self-consistency (batch vs single, argmax
echo) and by *human-coherence* checks, never logit-vs-HF. QK-norm-off
attention still produces plausible text on short prompts, which is why the
Entry 58–66 30B generation hunt ("repetition, argmax stuck") never found
that the QK-norm flag was silently zero. The fix adds a native-vs-HF
harness (`arch/verify.py :: verify_native`) so every arch switch is
measured, exactly like the Python path (Entry 28).

## 5. Fixes landed (2026-08-31, Entry 67)

1. **decode_kernel.c**
   - `KernelCfg`: `qk_norm` → `qk_norm_type` (0 none / 1 per_head / 2 whole);
     added `bias_qkv`, `norm_topk_prob`. `KernelW`: added `bq/bk/bv` (fp32).
   - Attention: QKV bias add; QK-norm whole (pre-head-view, OLMoE) and
     per_head (post-view, Qwen3) both implemented.
   - MoE routing: top-k renormalization gated by `norm_topk_prob`.
   - Workspace: heap-allocated **separate** buffers sized from KernelCfg,
     cached by a shape signature (realloc on model switch), no offset math
     (the aliasing that broke the Entry 55 attempt). B capped at 16 to match.
   - `prefill_layers`: lm_head computed only at the last token (`logits=NULL`
     guard) — matches the docstring, saves T−1 lm_head reads.
   - `q4_dot_test` uses runtime E/m (set at ws_init).
2. **decoder3.py** — `NativeDecoder` now builds its `KernelCfg` from the
   registry (`get_spec`), not ad-hoc config reads: qk_norm_type, bias,
   norm_topk_prob, router naming (`mlp.gate.weight` vs
   `mlp.block_sparse_moe.router.weight`), and RoPE inv_freq via the same
   `llama3_inv_freq` code the Python path uses. Capability gate: raises for
   `mla` (deepseek) and gemma/phi/gpt_oss.
3. **generic_streamer.py** — same capability gate; `_qk_rms` uses `spec.eps`
   (was hardcoded 1e-6; olmoe registry default is 1e-5).
4. **arch/verify.py** — `verify_native(model_dir)`: NativeDecoder prefill
   logits + greedy vs HF, same PASS criteria as the Python harness.
5. Verified on Llama-3.2-1B, Qwen2.5-1.5B, SmolLM2-1.7B, OLMoE-1B-7B (see
   VALIDATION_LOG Entry 67 for numbers).

## 6. Still open (honest)

- **Mixtral** native: implemented in code (router naming + no-renorm) but
  not verified (no weights on disk) — the harness will gate it when
  downloaded.
- **Sliding-window attention** (Qwen2.5 long contexts): not modeled; only
  matters past the window (>32k/128k tokens).
- **DeepSeek MLA / gemma / phi / gpt_oss**: detection + conversion only;
  engine raises until the math lands.
