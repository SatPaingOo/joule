# Joule — Journey Documentation (v0.4 Database-Style Inference)

> 2026-08-29/30 | Full journey from paradigm validation to a working product loop.
> Detailed experiment data: [results/VALIDATION_LOG.md](../results/VALIDATION_LOG.md) (Entries 16-73).
> The retrospective: [docs/JOULE_PAPER.md](JOULE_PAPER.md).

---

## 1. Vision

**"AI models work like databases."** Store weights on disk, query on demand, load only
what a query needs, release after use — with input/output tokens and quality identical
to the original model, while power and resource usage collapse.

Master equation: `π*(x) = argmin E[T(π,x)] s.t. V(π,x) = PASS`

## 2. What the model actually is (Entry: MODEL_ANATOMY)

A model file is a named tensor database. FFN = 77% of weights and is naturally sparse:
per query only ~50% of neurons carry 90% of activation mass. safetensors mmap gives
row-level access (neuron = 28 KB row, <0.1 ms). The paradigm is not an analogy — the
file format already supports it.

## 3. The validation ladder (what was proven, in order)

| Phase | Proof | Key number |
|---|---|---|
| A (Entry 16) | Dense masked serving works; prompt encoding must stay exact; answer cache | 7B FFN keep 51-52%; cache 0.0003 ms |
| B (Entry 17) | True row-streaming engine; load→serve→release | pool = 52% of FFN bytes; masked speed = full speed |
| C (Entry 18) | Adaptive refresh + margin-tolerant verify gate | fallback 2/3 → 0/3; system lossless always |
| D (Entry 19) | Trained probes (closed-form ridge, 60 s for 7B) | mask IoU 0.82-0.84 linear-only |
| E (Entry 20) | MoE machinery: expert store + LRU pool | thrash cliff measured; LRU rescues (0%→85% hit) |
| **F (Entry 21)** | **Real MoE (OLMoE): streaming is budget-invariant** | **sparse(4GB) ≡ full(11.4GB), token-identical 3/3** |
| **F (Entry 22)** | **Frontier 30B (61 GB) > machine RAM (31 GB) — served anyway** | outputs budget-independent 3/3 |
| G (Entry 23) | Q4 expert store (61→15.4 GB) | IO/token ÷19-30; answers identical to bf16 |
| **G (Entry 24)** | **Product loop: joule convert + joule serve (OpenAI API)** | 61 GB model served over HTTP on a laptop |
| H (Entry 25-26) | Speed stack v1: C kernels + AVX2 + fused prefill | prefill 155s→13.5s (11×); decode ~5.3 tok/s |
| H (Entry 27) | Spec decode (cross-family draft) | **negative result**: acceptance 8-11% — needs same-family draft |

## 4. Core discoveries (new knowledge produced)

1. **Prompt encoding must be full-precision; selective loading is decode-time policy.**
   Masking prefill collapses generation (both 1.5B and 7B measured).
2. **For real MoE models, router-guided streaming is mathematically lossless** — the
   model defines its own sparse compute; Joule only changes WHERE weights live.
3. **The thrash cliff**: naive mmap (llama.cpp-style) collapses when the hot working set
   exceeds RAM; router-guided LRU avoids it — smart scheduling beats raw hardware.
4. **MoE router = free mask**: for MoE targets, the model hands us the selectivity map;
   dense models need probes (which train in ~1 minute on CPU at 0.82+ IoU).
5. **Converter's job is now concrete**: re-layout weights for neuron/expert-granular
   access (down_t / Q4 store) + train probes + emit a manifest/report card.
6. **Cross-family speculative decoding fails** (8-11% acceptance); same-family draft
   required — but 0.6B (Entry 69) also gave acceptance ~0: the draft must be close
   in size (1.7B/4B for a 30B target).

## 5. Product shape (works today)

```
joule convert <model>   → arch detect + Q4 store + probes + manifest + report card
joule serve <model>     → OpenAI-compatible API (streaming), persistent answer cache,
                          /status (RAM pool, cache hits, tok/s)
```
Architecture registry: qwen2 (dense masked), qwen3 (dense), olmoe + qwen3_moe
(expert streaming). Native stack: zig-compiled C kernels (fused Q4-dequant GEMV/GEMM,
expert_job), threaded via Python executor, zero CRT dependency.

## 6. Current numbers (Qwen3-30B-A3B on a 31 GB Ryzen AI laptop)

| Metric | Entry 22 (start) | Now (Entries 26-42) |
| *(final, Entry 50/70)* | — | single-stream ~7-10 tok/s floor; batch kernel 19.6 @ B=8 (dummy); real serve ~3-5 tok/s |
|---|---|---|
| First token (prefill) | ~155 s | **20 s** (target < 10 s) |
| Decode single-stream (native batch kernel, B=1) | ~0.1 tok/s mixed | **~4.9 tok/s** |
| Decode aggregate (batch B=4-6) | — | **~10 tok/s** |
| Disk | 61 GB bf16 | **15.4 GB Q4** |
| RAM | impossible to load | **fixed 1.87 GB + working set 8-13 GB** |
| Quality (MoE) | — | **budget-invariant** (identical across RAM budgets; Q4 long-gen drifts, Entry 73) |
| Batch kernel correctness | — | **B=1 == single bit-identical; B=2-4 ≤ 7e-7** (Entry 42) |

## 6.5 Layer-skip reality check (Entry 43 — what "selective" really means)

Measured on Qwen3-8B (`layer_skip_probe.py`): whole-layer skipping via the
residual stream is valid math, but only **~11% of layers are skip-safe**
(single-skip logit drift < 0.5), and influence is **model-inherent, not
query-dependent** — the same layers are low-influence for every input. So
per-query layer selection is NOT viable. The real selectivity levers:
- **MoE top-k** (model-defined, exact) — the current product path
- **FFN neuron sparsity** (~50% inactive, verify-gated approximate) —
  Joule's phase B/C/D prototypes; the real differentiator

## 7. Honest gaps (to "normal user daily-usable")

1. First token still ~20 s cold (warm conversation pool + prefill trimming queued).
2. Single-stream decode ~7-10 → target 10-20+ (close-in-size spec decode, int8 VNNI GEMM, NPU).
3. 50+ tok/s on THIS hardware class is CPU-bandwidth-bound — that tier belongs to
   high-bandwidth devices (Strix Halo / Mac / GPU), where the same stack scales.
4. Aggregate throughput requires the int8 GEMM micro-kernel + batch kernel wired into serve
   (kernel built + correctness-verified; server wiring DONE in Entry 68/70 —
   real serve aggregate ~3-5 tok/s).
5. Generic converter (any arch: Kimi/GLM/DeepSeek) + shape-generic kernel pending.
6. Benchmark at scale (200+ queries, lm-eval-harness) for third-party credibility.
7. Multi-turn, long-context, agent strict mode — untested.

## 8. Artifacts map

- Engine: `src/jouleai/engine/` (generic_streamer, stream_engine, masked_mlp),
  `storage/` (weight_store, expert_store, q4_store), `native/` (C kernels +
  wrappers + batch_bench/batch_correctness_test), `routing/` (policies, probes),
  `cli/` (joule_convert, joule_serve), `monitor/`, `experiments/`
  (phase A-F scripts + layer_skip_probe)
- Data: `results/` (validation JSONs), [results/VALIDATION_LOG.md](../results/VALIDATION_LOG.md) (Entries 16-73),
  `storage/converted/<model>/` (Q4 stores, probes)
- Docs: `docs/MODEL_ANATOMY.md`, `docs/BATCH_DECODER.md`, `docs/USAGE.md`, this file

---

## 9. The native-kernel era (Entries 28-73) — correctness, then the floor

After the Python prototypes (phases A-F), the project moved to a **native C
kernel** (one call per token, registry-driven arch-genericity) and then hit the
**long-generation drift** — the session that produced this paper's negative
results.

| Entry range | What happened |
|---|---|
| 28-31 | ArchRegistry (11 families), path-only serve across 2 MoE families, browser chat E2E |
| 32-42 | Native kernel3: full decode in C, batch decode kernel, ggml-exact spin barrier, ALL-PASS correctness |
| 43-48 | Layer-skip probe (NOT viable, §6.2), int8 VNNI attention (6× single-stream), VNNI Q4 FFN wash |
| 49-52 | C prefill + native serve, mmap-lazy RAM ∝ working set, fixed-weight cache (startup 22→3.4 s) |
| 53-57 | Native batch serve (aggregate scales with B), spec decode fails cross-family, shape-generic kernel |
| 58-66 | The 30B repetition hunt — last-layer FFN collapse → workspace aliasing → heap-workspace fix |
| 67-70 | Registry-driven arch-generic kernel verified on 5 models; batch serve + argmax decode |
| 71-73 | **i8 expert tier reverted** (2× slower); drift hunt: KV correct, then **quantization root-caused**; **bf16-exact expert tier** (reps=0) |

## 10. The honest end-state (this paper's verdict)

- **What Joule proved**: a 61 GB MoE served on 31 GB RAM (budget-invariant —
  identical output at different RAM budgets), RAM ∝ working set; a correct,
  arch-generic native kernel verified on 5 models; kernel batch best case
  19.6 tok/s @ B=8 (dummy tokens; real serve aggregate ~3-5 tok/s, Entry 70).
- **What Joule disproved**: "fast single-stream MoE chat on a laptop" (hardware
  floor ~7-10 tok/s), inference-only layer skipping (quality collapses), expert
  output caching (input-dependent), cross-family spec decode (draft mismatch).
- **The quality fix**: bf16-exact expert tier eliminates long-gen repetition
  (64 tokens, reps=0 vs Q4's reps=10) — quantization error at the router
  threshold is the drift cause.

Full detail: [docs/JOULE_PAPER.md](JOULE_PAPER.md) (the retrospective), [results/VALIDATION_LOG.md](../results/VALIDATION_LOG.md)
(Entries 16-73).
