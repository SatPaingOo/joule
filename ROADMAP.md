# Joule — Roadmap (current, 2026-09)

> **Status**: v0.4 working prototype is done (see [README](README.md) and
> [results/VALIDATION_LOG.md](results/VALIDATION_LOG.md) Entries 16-73).
> This roadmap is the *current* plan — what's proven, what's next, what's
> hardware-gated, and what was measured to be not worth doing. (The original
> pre-kernel roadmap is superseded; the historical record lives in
> [docs/JOURNEY.md](docs/JOURNEY.md).)

---

## ✅ Done (v0.4 — measured, verified)

| Area | State | Evidence |
|---|---|---|
| Disk-backed MoE serving (61 GB on 31 GB RAM) | working, RAM ∝ working set, budget-invariant | Entry 22/51 |
| Registry-driven arch-generic native C kernel | verified vs HF on 5 models (qwen2/qwen3/llama/mistral/olmoe/qwen3_moe) | Entry 67 |
| Batch decode kernel (spin barrier, Q4 batch GEMM) | correct (B=1 bit-identical, B=2-4 ≤ 7e-7); best case 19.6 tok/s @ B=8 | Entry 42/48 |
| OpenAI-compatible server + browser chat | works (MoE models); batched prefill ~2 s first token | Entry 68 |
| Expert tiers | Q4 (default), bf16-exact (fixes long-gen drift), i8 (built, reverted) | Entry 71/73 |
| Verify harness (`verify.py --native`) | auto-PASS gate per arch family | Entry 67 |
| Docs + evidence | retrospective, FAQ (20 Q), arch audit, VALIDATION_LOG committed | — |

---

## 🔜 Next (ranked by value ÷ effort)

1. **GPU / Strix-Halo / Mac validation** — run the same kernel on a
   bandwidth-rich device. The 150-480 tok/s figures are projections, not
   measurements; this is the single unmeasured claim, and the paradigm's best
   case lives there. (Control plane already adapts per device.)
2. **Comparative benchmark vs Ollama / llama.cpp** — same model, same prompts,
   wall-clock + peak RAM + tok/s. Turns "why not just use Ollama" into data.
3. **Calibrated quantization on the 30B experts (GPTQ/AWQ-style)** — the
   router-threshold finding predicts *reducing average error is not enough*;
   test whether calibrated quantization crosses the router's decision boundary
   and kills the long-gen drift.
4. **Long-horizon verification (64+ tokens) in the harness** — the ≤12-token
   horizon is why the Q4 drift hid for two sessions (Entry 71-73).
5. **Dense serve wiring** — the kernel's dense path is verified; the serve
   wiring is the remaining product gap.
6. **Linux/macOS port of the native kernels** — the biggest adoption unlock
   (currently Windows x64; the pure-Python path runs anywhere but is slow).
7. **More arch families** — deepseek (MLA), gemma, phi, gpt_oss are detected +
   gated with a loud error; the registry + verify harness makes each testable
   in one command once the kernel math lands.
8. **Speculative decoding with a close-in-size draft** — 0.6B rejected
   (Entry 69 acceptance ~0); a 1.7B/4B draft for the 30B target is the next
   try, per the same-family-close-in-size requirement.
9. **int8 VNNI GEMM for attention + lm_head** — the remaining compute-bound
   bottleneck at B>1 (per-step time grows with B); converts compute-bound into
   bandwidth at batch.
10. **Third-party credibility benchmark** — 200+ queries / lm-eval-harness.

---

## ⏳ Deferred (hardware / traffic gated)

| Item | Trigger |
|---|---|
| GPU validation (item 1 above) | when a bandwidth-rich device is available |
| Vector DB upgrade for the answer cache | when cache > 10k entries |
| 200+ query calibration data | paper phase / production traffic |
| Tier B formal proof of the verify gate | paper phase |
| NPU backend | when an NPU device is available |

---

## ❌ Disproven — do not revisit (measured, this repo)

These were the project's own research threads; each was tested and the
negative result is documented. Anyone tempted to try them should read the
corresponding section first:

| Thread | Why it fails | Evidence |
|---|---|---|
| Per-query layer selection ("this query needs layers 1,2,3,25") | layer influence is model-inherent, not query-dependent | Entry 43, JOULE_PAPER §6.2 |
| Inference-only layer skipping / early exit | KV-coupling + no redundancy in shallow MoE; only *trained* skipping (MoD/LayerSkip) works | §6.2 |
| Expert-output cache | expert outputs are input-dependent; the ceiling is bandwidth, not disk IO | §6.3 |
| Cross-family speculative decoding | tokenizer + distribution mismatch → acceptance ~0-1% | Entry 54/69 |
| i8 expert tier | 2× bytes → 2× bandwidth, quantize overhead eats VNNI at tiny row-dots | Entry 71 |
| Adaptive FFN neuron masking on CPU | probe + delta-gather + pool rebuild cost > FFN compute saved | Entry 19 |

---

## How to contribute

Any "Next" item is a good starting point. See [docs/STANDARDS.md](docs/STANDARDS.md)
for the quality gates (verify PASS, semantic naming, no version suffixes) and
[docs/USAGE.md](docs/USAGE.md) for running. Ranked contribution values:
Linux/macOS port > benchmarks > calibrated quantization > dense wiring.
