# Joule: A Database-Style Inference Engine for Memory-Constrained MoE Serving

## An Engineering Retrospective — What Worked, What Failed, and the Hardware Floor

**Status**: Retrospective of the Joule project (v0.4, 2026-08-29 → 2026-08-31).
**Models**: Qwen3-30B-A3B, Qwen3-8B, OLMoE-1B-7B, Llama-3.2-1B, Qwen2.5-1.5B, SmolLM2-1.7B (all verified).
**Hardware**: Ryzen AI 7 350, 8C/16T, 31 GB RAM, ~35 GB/s effective memory bandwidth.
**Companion**: [results/VALIDATION_LOG.md](../results/VALIDATION_LOG.md) (Entries 16–73), [docs/JOURNEY.md](JOURNEY.md).

---

## Abstract

Joule is a database-style inference engine for serving mixture-of-experts (MoE)
large language models on memory-constrained hardware. Instead of loading model
weights into RAM, weights live on disk and are selectively loaded per query —
"load only what you need, compute only what you use, release after use."

This retrospective documents the full engineering journey: what was built, what
was measured, what worked, and — most importantly — what **failed** and why.
The central finding is a hardware truth: **on a 31 GB DDR laptop, single-stream
MoE decode is memory-bandwidth-bound at ~7–10 tok/s, and no software
optimization escapes this floor.** The project's flagship claim — serving a
61 GB frontier-family MoE (Qwen3-30B-A3B) on a 31 GB laptop — was achieved and
verified **budget-invariant** (identical output at different RAM budgets).
But the "30–150 tok/s" speed target was shown to be
unreachable on this hardware for single-stream decode, and reachable only as a
**batch aggregate** across concurrent sessions.

The paper's value is in the negative results: (1) long-generation drift in MoE
models is **quantization accumulation**, not a kernel bug; (2) inference-time
layer/FFN skipping **does not preserve quality** on shallow MoE models; (3) an
"expert-output cache" is **fundamentally impossible** because expert outputs are
input-dependent; (4) cross-family speculative decoding fails; (5) the
"30–150 tok/s" target is a physics statement, not an engineering one.

---

## 1. Motivation

### 1.1 The problem

Frontier LLMs are too large for consumer hardware. A 30B-parameter MoE in bf16
is ~61 GB — it cannot be loaded into a 31 GB laptop's RAM by any conventional
runtime (Ollama, llama.cpp, HF transformers all segfault or thrash). Yet MoE
models have a special property: only a small **active set** of parameters is
used per token (e.g., Qwen3-30B-A3B uses 3.3B of 30.5B active — top-8 of 128
experts per layer).

### 1.2 The hypothesis

If weights live on disk and are loaded **on demand** (expert-by-expert), then:
- RAM usage ∝ working set, not model size
- A model larger than RAM can be served
- With smart caching, disk IO becomes negligible and the system is
  compute/bandwidth-bound at the *active set*, not the full model

### 1.3 The vision (as originally stated)

> "A question should use as much depth as it needs — easy questions can answer
> at layer 12, hard ones need all 36."

This vision drove two research threads: (a) selective **weight** loading
(expert streaming — worked), and (b) selective **computation** (layer skipping,
early exit, neuron masking — partially failed, see §6).

---

## 2. Architecture

### 2.1 Storage layer (SenseWeightStore)

Weights are stored as named tensors on disk, mmap'd lazily. A converter
(`joule convert`) re-lays-out the `down_proj` columns and builds a Q4 expert
store. The key design: **zero-copy row access** — `rows(name, idx)` touches
only the requested rows' bytes, so memory is faulted in on first access.

### 2.2 Expert store tiers

| Tier | Size (30B) | Error | Speed | Status |
|---|---|---|---|---|
| bf16 (original) | 61 GB | exact | reference | doesn't fit |
| **Q4** (group-64 int4, fp16 scales) | 15.4 GB | ~11% rms | default | **shipped** |
| int8 (Q8_0, per-row fp32 scale) | 29.1 GB | ~1.5% rms | 2× slower | reverted (Entry 71) |
| bf16-exact tier (new, this session) | 61 GB | exact | 2× slower than Q4 | **fits OLMoE only** |

### 2.3 Native C kernel

A dependency-free (nostdlib, zig-compiled) Windows DLL with:
- Fused Q4-dequant GEMV (`q4_gemv`), AVX2 batched row-dots
- Full 48-layer decode (`decode_layers_batch`) — route + experts + attention
  (QK-norm, RoPE, SDPA) + lm_head, one C call per token
- ggml-exact spin barrier thread pool (fixed worker ids, monotonic counters)
- **Registry-driven arch-genericity**: one kernel, flags from config.json
  (qk_norm type, bias_qkv, norm_topk_prob, rope scaling, dense/MoE)

### 2.4 The registry

One `get_spec(config.json)` produces an `ArchSpec` covering 11 families:
qwen2/qwen3/llama/mistral/gemma/phi/gpt_oss (dense) + olmoe/qwen3_moe/mixtral/
deepseek (MoE). Verified native-vs-HF on 5 models (Entry 67):

| Model | Arch | max\|dlogit\| | Greedy identical | Verdict |
|---|---|---|---|---|
| Llama-3.2-1B | llama + llama3 rope | 0.19 / 0.16 | identical | **PASS** |
| Qwen2.5-1.5B | qwen2 (bias) | 0.35 / 0.33 | identical | **PASS** |
| SmolLM2-1.7B | llama (θ=130k) | 0.23 / 0.22 | identical | **PASS** |
| Qwen3-8B | qwen3 (per-head QK-norm) | 0.64 / 0.53 | identical | **PASS** |
| OLMoE-1B-7B | olmoe (whole QK-norm) | 2.88 / 1.59 | identical | **PASS** |

---

## 3. What Worked (Measured Results)

### 3.1 Serving a model larger than RAM (budget-invariant)

The flagship claim. Qwen3-30B-A3B (61 GB bf16, 15.4 GB Q4) served on a 31 GB
laptop with an 8 GB pool budget:

- **RAM ∝ working set**: 8.3 GB RSS after load (fixed weights only), ~13.4 GB
  after 40 decode tokens (experts page in as used) — vs 61 GB full load.
- **Budget-independent outputs**: runs at 8 GB and 16 GB pool budgets produced
  **identical token streams** (Entry 22) — budget-invariant. Note: this is
  self-consistency (same engine, different budget), NOT a HF-reference
  check — the 30B's HF-bf16 reference cannot load on this machine
  (61 GB, Entry 71).
- **Startup 3.4 s** (fixed-weight .npy cache, Entry 52) from 22 s.

### 3.2 The native kernel is correct

Every component verified against HF bf16 to corr ≥ 0.999:
attention, QK-norm (whole + per-head), RoPE (theta + llama3 scaling), routing,
expert FFN, KV cache. Batch decode B=2/3/4 ≤ 7e-7 of B=1 (bit-identical at
B=1). This correctness was hard-won — see the debugging ladder in §5.

### 3.3 Speed ladder (single-stream, 30B, Q4)

| Step | B=1 tok/s | B=8 aggregate | Fixed RAM |
|---|---|---|---|
| Session start (fp32, no batch) | 1.5 | — | 6.2 GB |
| + batch kernel + ggml barrier | 4.9 | 8.5 | 6.2 |
| + bf16 fixed weights | 4.2 | ~15 | 3.1 |
| + int8 attention (VNNI) | **9.4** | **19.4** | 1.87 |
| + VNNI Q4 FFN (wash) | 9.3 | 19.6 | 1.87 |

**6× single-stream** (1.5 → 9.4 tok/s) and **batch aggregate 19.6 tok/s @ B=8**
— the batch amortization (weights read once per B tokens) is the real speed
lever, and it scales with B.

> **Caveat (honest)**: the 19.6 tok/s @ B=8 is the **standalone kernel
> benchmark with dummy tokens whose routing collides** (fewer unique
> experts/layer → fewer Q4 row-dots). Real prompts route to more unique
> experts; the measured real serve aggregate is **3.1 words/s** (≈4–5 tok/s,
> Entry 70). 19.6 is the kernel's best case, not the product number.

### 3.4 Productized end-to-end

`joule convert` → `joule serve` → OpenAI-compatible HTTP + browser chat:
- First token 155 s → 20 s (fused AVX2 prefill, Entry 26) → ~2 s
  (batched prefill, Entry 68 — 10.6× first-token speedup)
- Repeat queries served from persistent answer cache in ~17.8 ms
- Any model path works: switch 9 models via `GET /v1/model/<name>`

---

## 4. The Hardware Floor (Physics, Not Engineering)

The single most important quantitative finding. On this 31 GB DDR laptop
(~35 GB/s effective bandwidth):

**Per decode token, the kernel reads ~3–4 GB of weights** (attention + lm_head
+ top-8 experts' Q4 weights). Therefore:

```
4 GB ÷ 35 GB/s ≈ 8–9 tok/s
```

This matches the measured 7–10 tok/s exactly. The floor is **memory bandwidth**,
not compute, not software quality.

| Target | Required bandwidth | Feasible on this laptop? |
|---|---|---|
| 10 tok/s | ~35 GB/s | ✅ (the floor) |
| 30 tok/s | ~120 GB/s | ❌ (3.4× more) |
| 150 tok/s | ~600 GB/s | ❌ (17× more) |

The 30–150 tok/s target is reachable only as a **batch aggregate** (B sessions
decode together, weights read once per B — kernel best case 19.6 tok/s @ B=8
with dummy tokens; real serve aggregate ~3–5 tok/s, Entry 70) or on
high-bandwidth devices (Apple Silicon / Strix Halo / GPU).

**Implication**: any project claiming "fast single-stream MoE chat on a laptop"
is either (a) using a much smaller model, or (b) wrong about the hardware.

---

## 5. The Debugging Ladder (How Correctness Was Hard-Won)

A catalog of real bugs found through systematic isolation — useful as a
checklist for anyone building a similar kernel.

### 5.1 The 30B generation bug (Entries 58–66)

Symptom: repetition / garbage on long generation. The hunt:
- Layer 3 FFN "corruption" → actually the **last-layer FFN collapse**
- `decode_layers_batch ≠ debug_hidden_n` → a **state bug in the real path**
- L27 FFN input 2× too large → traced to the **last-layer attention output**

The root cause was a **workspace aliasing bug** in the static-buffer kernel:
`ws->h`/`ws->tmp`/`ws->h2` overlapped the FFN's act/y regions at specific
shapes. Fixed by the **heap workspace with separate mallocs** (no offset math)
— the shape-generic refactor (Entry 55).

### 5.2 The stale-DLL trap (Entry 55)

`build_native.py`'s `exists()` check passed with a **stale cached DLL**,
hiding real compile errors for many edits. Forced clean rebuild revealed them.
**Lesson**: never trust a build tool's existence check; force clean rebuilds.

### 5.3 The ctypes struct mismatch (Entry 42)

`KernelCfg` C struct had an `intermediate` field the Python ctypes struct never
declared — ctypes silently dropped the kwarg and C read garbage as the FFN
intermediate size, driving expert loops into wild memory → segfault at B=1/L=1.
**Lesson**: keep the ctypes struct byte-for-byte in sync with C.

### 5.4 The i8 batch race (found this session)

The i8 expert FFN worker quantized its activations into a **shared `ws->aq`
buffer**; with the thread pool (B>1), concurrent workers overwrote each other's
rows → garbage on seq 1+ in every batch decode. Fixed by slicing `ws->aq` per
pool participant. **Lesson**: any shared scratch buffer written by parallel
workers is a race; make it per-worker.

### 5.5 Other real bugs (condensed)

- QK-norm applied after head view instead of before (OLMoE whole-vector)
- llama3 rope scaling: multiplying instead of dividing inv_freq
- zeros-FFN (union-batched results never index_add'ed)
- nibble mask wrong order in AVX2 (micro-test passed, full-gen caught it)
- `var history = []` shadowing `window.history` in the browser UI
- spin barrier: arrival-order worker ids + main not joining → segfault (×2)

---

## 6. Negative Results (The Paper's Real Value)

### 6.1 Long-generation drift = quantization accumulation (Entries 71–73)

**Symptom**: 64-token generations drift into repetition/garbage; 32-token
answers are clean. Divergence at token ~7–16.

**The hunt** (this session): systematically verified every component vs HF:
- Prefill KV: corr 1.0 (Entry 72)
- Decode KV: flat error ~1.0 maxdiff across all positions — no jump, no state bug
- expf_fast softmax: 3e-8 error, same top-8
- Router: healthy (diverse expert sets, corr 0.999+)
- Per-expert FFN: correct to bf16 rounding

**The verdict**: with **exact bf16 experts**, OLMoE produces 64 **coherent,
zero-repetition** tokens ("The capital of France is Paris. Paris is the largest
city..."), while Q4 experts produce 10 repeated 4-grams. The drift is **Q4
quantization error accumulating through the hypersensitive last layer** — not a
kernel bug. i8 (1.5% error) also flips at step 7 (different error profile).
Entry 71's "q4==i8 for 20 tokens" was misleading: both diverge at the *same
step* (7), looking identical, but that's a coincidence of the error profile.

**Fix**: a bf16-exact expert tier (`precision="bf16"`) — removes quantization
entirely. Verified: 64 tokens, reps=0 (vs q4 reps=10). Tradeoff: 2× the Q4
IO (3.1 vs 7–10 tok/s on OLMoE).

**Caveat for the 30B**: bf16 experts = 61 GB, does not fit. Q4/i8 both repeat
on echo-prone prompts ("Human:/Assistant:" loop) — but this is
quantization-triggered echo-locking, not a kernel bug: the 30B is deterministic,
routing diverse, KV healthy, and prompt 1 ("photosynthesis") produces clean 64
tokens.

### 6.2 Inference-time layer skipping does NOT preserve quality (probes + this session)

**The vision**: "compute only the layers the input needs." Tested on:

| Model | Skip | Speedup | Quality |
|---|---|---|---|
| Qwen3-8B (36L dense) | last 40% low-influence | ~1.6× | first-token argmax 4/4 |
| OLMoE (16L MoE) | first 4 layers | 1.8× | **collapse** (58 reps) |
| OLMoE | last 4 layers | 1.5× | **collapse** |
| OLMoE | FFN-skip first 4 (KV kept) | 1.3× | **gibberish** |
| 30B (48L MoE) | last 6 | 1.8× | no-repeat but **gibberish** |

**Why it fails**:
1. **KV coupling**: skipping a layer removes its KV, which later layers'
   attention needs — layer-skip breaks attention state.
2. **Shallow MoE has no redundancy**: Qwen3-8B (36L) has a skip-safe tail;
   OLMoE (16L) has *every layer critical*.
3. **Block influence ≠ skip-safe**: OLMoE's first layers have low block
   influence but skipping them collapses output.

**Conclusion**: MoD and LayerSkip (the training-based methods) work because
they *train* the model to tolerate skipping. Inference-only skipping on
untrained models fails on shallow/deep MoE alike.

### 6.3 Expert-output cache is fundamentally impossible

**The idea**: consecutive tokens route to overlapping experts, so cache the
expert outputs and reuse them.

**Measured**: consecutive tokens share 2.7/8 experts (34%); a 6-step cache
would cover 74%.

**Why it fails**: expert outputs are **input-dependent** — token t's expert
output depends on token t's `ffn_in`, which differs from token t-1's. You
cannot reuse the output; you can only reuse the *weights* (which are already
page-cached by mmap). The "cache" reduces disk IO, not bandwidth — and
bandwidth is the floor.

### 6.4 Cross-family speculative decoding fails

Qwen2.5-1.5B drafting Qwen3-30B: acceptance 1% (Entry 54), 0.6B draft: ~0
(Entry 69). The draft must be **same-family and close in size** (the tokenizer
+ distribution must match). The verify machinery is correct; the draft is the
blocker.

### 6.5 FFN neuron masking is gather-bound on CPU

Masking ~50% of FFN neurons (Deja-Vu style) saved loading but the probe +
delta-gather + pool rebuild cost more than the FFN compute saved (Entry 19).
Static mask + margin-verify gate is the CPU sweet spot; adaptive masking is
only viable where gathers are cheap (GPU).

### 6.6 i8 expert tier is 2× slower

int8 experts (1.5% error) are 2× the Q4 bytes → 2× bandwidth, and the per-row
quantize overhead eats the VNNI win at tiny row-dots (Entry 71). Reverted.

---

## 7. What We Learned About Models (Transferable Knowledge)

Beyond Joule's own results, the project produced a body of knowledge about how
LLMs actually behave — the kind of "feel" for models, hardware, and kernels that
is rarely written down. This is the most transferable part of the project.

### 7.1 The residual stream is the real "state"

Every transformer layer reads and writes the **same residual vector**; attention
and FFN are deltas on top of it. This has practical consequences:
- Layer influence is **additive and small** (most layers change the hidden by
  ~0.1–0.5 on a ~10-norm stream) — but the *last* layer can be a huge transform
  (OLMoE L15: 6 → 97 in one layer).
- The model's "memory" lives in the KV cache + residual, not in any single
  layer — which is why skipping a layer breaks everything downstream (§6.2).

### 7.2 Every model family has its own "diff with llama" quirks

The registry-driven approach forced us to enumerate exactly how families differ.
These are the real differences (not marketing):

| Family | Quirk found |
|---|---|
| OLMoE | **whole-vector QK-norm** (RMS over hidden_size, before head view) — missing it → wrong outputs |
| Qwen3 | **per-head QK-norm** (after view, w=[hd]) |
| Llama-3.x | **llama3 rope scaling**: divide inv_freq by factor (multiplying is subtly wrong — 5-7 logit errors that still pass argmax) |
| Qwen2 | QKV **bias** |
| Qwen3-MoE | **norm_topk_prob=True** (renormalize top-k weights); OLMoE/Mixtral keep raw softmax |
| Mixtral/DeepSeek | **block_sparse_moe** expert naming (w1/w2/w3, not gate/up/down) |
| DeepSeek | **MLA** attention (latent KV) — a different attention architecture entirely |

**Lesson**: "one adapter per family, verified against HF" is the only safe way.
Coherence checks on short prompts **miss** arch-flag drift (the QK-norm and
llama3-rope bugs both passed short-prompt tests).

### 7.3 Quantization error is not uniform — and not the whole story

- Q4 (group-64 int4): ~11% rms on weights, but the *output* error is much
  smaller (per-dot errors ~0.005-0.013) because of error cancellation over
  2048-dim dots.
- The **router is the amplifier**: a 1% input error at the router's input can
  flip which 8 of 64 experts are selected — a discrete, discontinuous change.
- **i8 (1.5% error) can be WORSE than Q4 (11%) at a specific step** — the error
  profile matters more than the average. Both flipped at step 7, with different
  wrong tokens (margins 2.06 vs 17.5).
- bf16-exact experts are the only reliable fix for long generations — which is
  why "quantization is the problem" is the honest answer, even though i8-vs-Q4
  looked identical at the divergence step.

### 7.4 Position-dependence and "the 16-token mystery"

The drift appears at a **position** (~token 8-16), not at step 0. Why:
- Early tokens are "easy" — the model's confidence is high, logit margins large,
  so small errors don't flip argmax.
- As generation proceeds, the model enters lower-confidence territory (new
  content, repetition-prone), where logit margins shrink and the accumulated
  error crosses the flip threshold.
- **This is why short-prompt verification (≤12 tokens) passes while 64-token
  generation drifts** — the verify gate's horizon was too short.

### 7.5 KV cache is the most robust component

Across every model and every bug hunt, the KV cache was **never** the problem.
Prefill KV, decode KV, position tracking, GQA layout — all verified correct
repeatedly. This is worth knowing: if your MoE drifts, suspect the **FFN/route
path**, not attention. (The attention was the source in dense models' early
days; in MoE it's the experts.)

### 7.6 The lm_head is a hidden bottleneck

The lm_head (V×d) is the single biggest matvec — 311M MACs, ~1.2 GB read per
token for V=151k. It's ~40-50% of every decode step. Int8 quantization helps RAM
(1.87 GB) but the dequant ALU can be slower than the bandwidth saving (bf16
lm_head measured 37 GB/s vs fp32's 74 GB/s — dequant-bound). **The lm_head's
shape (vocab size) is often the real floor for small models.**

### 7.7 Threading and shared state: the kernel's dark corners

- **Spin barriers are delicate**: arrival-order worker ids + main-not-joining
  both segfaulted (×2 attempts). The ggml-exact pattern (fixed ids, monotonic
  counters) is the only correct one.
- **Shared scratch buffers are races**: the i8 `ws->aq` bug (this session)
  corrupted batch decode only at B>1. Any per-worker state must be per-worker.
- **ctypes structs must mirror C exactly**: one missing field → garbage bounds →
  segfault (Entry 42).
- **Build tools lie**: a stale cached DLL hid real compile errors (Entry 55).

### 7.8 Model scaling observations

- **Deeper models have redundancy; shallow ones don't**: Qwen3-8B (36L) had a
  skip-safe tail; OLMoE (16L) had none — every layer critical.
- **The last layer is special**: OLMoE's L15 is a ~15× amplifier (input ~6,
  output ~97). Small upstream errors become huge downstream flips. This is why
  "the last layer collapses" appeared in multiple bug hunts — it's the model's
  design, not a bug.
- **MoE active-set physics**: decode working set ≈ top-k × layers × expert
  bytes (30B: 8 × 48 × 9.4 MB ≈ 3.6 GB). Pools at 2-4× that stabilize hit
  rates; below it, thrash (0% LRU hit, 551 MB/token IO — Entry 20).

---

## 8. The Quantization-Quality Frontier

The session's most actionable finding: **MoE quality under quantization is
bounded by router sensitivity, not average error.**

- Q4: ~11% rms error — flips logits at step 7 (margin 2.06)
- i8: ~1.5% rms error — ALSO flips at step 7 (margin 17.5 — different profile)
- bf16: exact — 64 tokens coherent, reps=0

The router's discrete top-8 selection is the amplifier: a small input error
changes which experts are selected, and the hypersensitive last layer turns
that into repetition. **Reducing average quantization error does NOT
proportionally reduce drift** — the error must be reduced *below the router's
decision threshold*.

This explains the "Super Experts" research finding (pruning super experts →
repetition): the router's output distribution is what drives stability.

---

## 9. Lessons Learned (For Future Readers)

1. **Measure the bandwidth floor first.** Before building a fast-inference
   engine, compute bytes/token ÷ bandwidth. If the answer is below your target,
   no software optimization will help single-stream.

2. **Batch amortization is the only CPU speed lever for large models.**
   Weights read once per B tokens → aggregate scales with concurrency. The
   kernel-level batch (B=8, 19.6 tok/s) is the path, not single-stream tuning.

3. **MoE expert streaming is budget-invariant.** The model defines its
   own sparsity; you only change WHERE the weights live. Self-consistency
   (same engine, different budget → identical) held; reference-based
   losslessness is verified on the 5 small models, not the 30B.

4. **Quantization drift is a router-threshold problem, not an average-error
   problem.** Test long generations with exact weights before blaming the
   kernel.

5. **Per-component verification vs a reference is the only way to debug a
   kernel.** Every bug in this project was found by isolating a component
   against HF — never by staring at code.

6. **Negative results are results.** The layer-skip, expert-cache, spec-decode,
   and i8 findings each saved someone else from repeating the attempt.

7. **Build tools lie (stale DLLs).** Force clean rebuilds. Keep ctypes structs
   in sync with C. Never trust an `exists()` check.

8. **Models are more different than the marketing says.** "One architecture"
   hides real differences (QK-norm placement, rope scaling, router renorm,
   bias, expert naming). Always verify against a reference per family — short
   coherence checks are not enough.

9. **The last layer is where models hide their power and their fragility.**
   A hypersensitive final layer amplifies both the model's capability and any
   upstream numerical error. When debugging "collapse," check the last layer's
   input before its math.

10. **"Easy" and "hard" are not just about the question — they're about the
    model's confidence at that position.** Early tokens have large margins;
    later tokens have small ones. This is why drift appears at token 8-16, not
    token 0, and why verification must extend past the "easy" horizon.

11. **The best debugging tool is a good reference implementation.** HF
    transformers, run step-by-step with hooks, was the oracle that resolved
    every dispute in this project. Invest in the comparison harness before the
    kernel.

---

## 10. What Joule Is (Honest Positioning)

Joule is **not** a competitor to Ollama/llama.cpp for fast chat on a laptop —
the hardware floor guarantees parity, not superiority.

Joule **is** a working demonstration of:
- Serving a model larger than RAM (budget-invariant), RAM ∝ working set
- A registry-driven, arch-generic native kernel verified against HF on 5 models
- Batch amortization as the CPU speed path (kernel best case 19.6 tok/s @ B=8
  with dummy tokens; real serve aggregate ~3-5 tok/s, Entry 70)
- A complete product loop (convert → serve → HTTP → browser chat)

**Where Joule would be genuinely useful**: memory-constrained edge devices
(8–16 GB mini PCs), batch/multi-user serving (where the aggregate matters), and
as a reference for "database-style" inference design.

---

## 11. Appendix: Key Numbers

| Metric | Value |
|---|---|
| 30B Q4 store | 15.4 GB (61 GB bf16 → ÷4) |
| Single-stream decode (30B, Q4) | ~7–10 tok/s (bandwidth floor) |
| Batch aggregate (B=8, kernel bench) | 19.6 tok/s (dummy tokens; real serve ≈ 3–5 tok/s) |
| Startup (cached) | 3.4 s |
| First token (batched prefill) | ~2 s (cold ~20 s) |
| Verified models | 5 (native-vs-HF PASS) |
| Arch families detected | 11 |
| Kernel DLLs (nostdlib) | quant_gemv 3 KB, expert_ffn 7.5 KB, decode_kernel 123 KB |
| Q4 expert error | ~11% rms |
| i8 expert error | ~1.5% rms |
| OLMoE bf16 64-token repetition | 0 (vs Q4: 10) |
| Expert overlap consecutive tokens | 34% (2.7/8) |
| RAM ∝ working set | 8.3 GB load → 13.4 GB after 40 tokens (vs 61 GB) |

---

*This document is the engineering record of what Joule tried, built, measured,
and learned. It is written to be useful to anyone building inference engines for
memory-constrained hardware — including the things that didn't work, which are
often the most instructive.*
