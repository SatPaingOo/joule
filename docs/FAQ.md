# Joule — Anticipated Questions (FAQ)

> Honest answers to the questions this repo is likely to attract.
> Claims trace to [results/VALIDATION_LOG.md](../results/VALIDATION_LOG.md)
> (Entries 16-73) unless marked as opinion.

## "Isn't this just re-doing research that already exists?"

Partly — and the docs say so. The distinguishing content is **measurement,
not novelty of direction**. What was known vs what Joule measured:

| Already known (literature) | What Joule measured (this repo) |
|---|---|
| Contextual sparsity exists in FFNs (Deja Vu, 2023) | Neuron activation measured per layer on Qwen2.5-1.5B: boundary layers ~12-30% active, middle ~52-70% (MODEL_ANATOMY §4) |
| Inference-only layer skipping degrades quality; trained skip (MoD/LayerSkip) works | The failure quantified on MoE specifically: OLMoE (16L) collapses on *any* skip, 30B (48L) gibbers on last-6; mechanism = KV-coupling + no redundancy in shallow MoE (JOULE_PAPER §6.2) |
| Expert outputs are input-dependent (can't be cached) | Quantified: consecutive tokens share only 2.7/8 experts (34%); a 6-step cache covers 74% and is still useless — the ceiling is bandwidth, not disk IO (§6.3) |
| Decode is memory-bandwidth-bound (llama.cpp #27478: 13.4 t/s on a Ryzen 9700X) | Independent measurement on this laptop: ~35 GB/s effective → 7-10 tok/s floor for 30B-A3B Q4 (§4) |
| Expert parallelism streams experts on GPU (DeepSeek-V2 EP) | Single-CPU disk-backed expert streaming: 61 GB MoE on 31 GB RAM, RAM ∝ working set (§3.1) |
| Quantization error compounds in MoE long generation | Root-caused to **router-threshold sensitivity**, not average error — i8 (1.5%) can be worse than Q4 (11%) at a given step; bf16-exact tier fixes it (§6.1, §7.3) |

The re-derivations are flagged as such (References §12 + "measured against, or
re-derived"). The original content — the registry-driven arch-generic kernel,
the debugging record, the RAM ∝ working-set measurement — is this repo's
contribution.

## "Why CPU-only? Why not GPU?"

No GPU was available (ROADMAP "Deferred"). The design is device-agnostic on
purpose:

- The control plane derives the execution plan from the device (tier,
  bandwidth, RAM) — docs/USAGE.md §4.0.
- The kernel math (attention, QK-norm, RoPE, routing, expert FFN) is verified
  against an HF reference — a device-independent oracle.
- The paradigm's **best case is on an accelerator or unified-memory device**:
  a 1 TB disk-backed MoE with a small RAM footprint is more compelling on a
  server than on a laptop.

**Explicit next step (not a design limitation)**: run the same kernel on a
bandwidth-rich device (Strix Halo / Mac / T4) and measure. The 150-480 tok/s
figures in this repo are projections for those devices, not measurements —
JOULE_PAPER §10.

## "Why is it so slow?"

Single-stream decode is memory-bandwidth-bound: ~4 GB of weights read per
token ÷ ~35 GB/s ≈ 7-10 tok/s. No software optimization escapes this on DDR.
The speed lever is batch amortization (weights read once per B tokens); the
measured real serve aggregate is ~3-5 tok/s, kernel best case 19.6 tok/s @ B=8
with dummy tokens (Entry 70). Full math: JOULE_PAPER §4.

## "Is it lossless?"

Two different claims, both documented:

- **Budget-invariant** — same engine at different RAM budgets, identical
  output (Entry 22). This is self-consistency, not a reference check.
- **Reference-verified** — native vs HF, greedy-identical on 2 short queries,
  on 5 small models (Entry 67). The 30B itself was never verified against HF:
  its 61 GB bf16 reference cannot load on this machine (Entry 71).
- Q4 long generations drift into repetition (quantization at the router
  threshold); the bf16-exact expert tier fixes it where the model fits
  (Entry 73).

So "lossless by construction" is wrong; "budget-invariant + short-horizon
reference-verified, with known Q4 long-gen drift" is right.

## "Why is serve MoE-only?"

The native serve path currently supports MoE models; dense serve wiring is next
(the kernel's dense path is verified, the wiring is not) — docs/USAGE.md §8.

## "Why Windows-only?"

The committed native DLLs are Windows x64 (built with zig, nostdlib). The
pure-Python path runs on any OS but is the slow prototype path. Porting the C
kernels to Linux/macOS is a build task, not a design barrier.

## "Is this usable as a product?"

No — and it doesn't claim to be. This is an engineering retrospective plus a
working technique demo (memory-constrained MoE serving). Ollama/llama.cpp are
better for fast laptop chat; the hardware floor guarantees Joule can't beat
them single-stream. Where it could be genuinely useful: 8-16 GB edge devices,
batch/multi-user serving, and as a reference design for disk-backed MoE.

## "What can I actually use from this repo?"

A few things, at different levels of effort:

- **As a reference for disk-backed MoE serving** — the storage layer, the
  Q4 expert store, the mmap-lazy loading with RAM ∝ working set. If you need
  to serve a model bigger than your RAM, this is a working design to copy.
- **As a catalog of arch differences** — the registry (docs/JOULE_PAPER.md
  §7.2) enumerates exactly how qwen2/qwen3/llama/olmoe/mixtral/deepseek differ
  (QK-norm placement, RoPE scaling, bias, router renormalization, expert
  naming). If you're writing a kernel or adapter for any of these, this table
  saves you the exact bugs it took Joule weeks to find.
- **As a debugging checklist** — the "debugging ladder" (§5): stale-DLL trap,
  ctypes struct drift, spin-barrier races, shared scratch buffers. If you're
  building a custom inference kernel, read §5 and §7.7 before you start.
- **As a measurement baseline** — the bandwidth-floor math (§4) and the
  quantization-router finding (§8): if you're evaluating MoE quantization,
  these are the numbers and the failure mode to design against.
- **To run chat on a memory-constrained machine** — the OpenAI-compatible
  server works today (MoE models, Windows). docs/USAGE.md.
- **To read as a cautionary engineering story** — what a paradigm-driven
  project looks like when the paradigm is wrong, including the part where the
  project disproves itself honestly.

## "What would be the highest-value next experiment?"

Ranked by value ÷ effort:

1. **GPU/Strix-Halo validation** — run the same kernel on a bandwidth-rich
   device. This is the single claim (150-480 tok/s projections) that is
   currently unmeasured, and the paradigm's best case lives there.
2. **Comparative benchmark vs Ollama/llama.cpp** — same model, same prompts,
   wall-clock + peak RAM + tokens/s. Directly answers "why not just use
   Ollama" with data instead of physics arguments.
3. **Calibrated quantization (GPTQ/AWQ-style) on the 30B experts** — the
   router-threshold finding predicts that *reducing average error is not
   enough*; test whether calibrated quantization actually crosses the router's
   decision boundary and kills the long-gen drift.
4. **Long-horizon verification (64+ tokens) in the harness** — the ≤12-token
   horizon is why the drift hid for two sessions (Entries 71-73).
5. **Dense serve wiring** — the kernel's dense path is verified; the serve
   wiring is the remaining product gap.

## "Can I contribute?"

Yes — the repo is MIT and structured for it. Highest-value contributions:

- **Port the native kernels to Linux/macOS** — the biggest adoption unlock
  (currently Windows x64).
- **Benchmarks** — Ollama/llama.cpp comparison, or GPU validation.
- **More arch families** — the registry + verify harness makes a new family
  testable in one command (`joule convert --verify` / `verify.py --native`).
- **Dense serve wiring** — roadmap item, kernel-side verified.

See docs/STANDARDS.md for the quality gates (verify harness PASS, no
version-suffixed names, semantic only) and docs/USAGE.md for how to run.

## "Why should I read the whole paper?"

The abstract and §4 (hardware floor) give the TL;DR. The value is in the
middle: §5 (debugging ladder — bugs you'll meet again), §6 (negative results —
layer skip, expert cache, spec decode, i8), §7 (what the measurements say
about how models actually behave). §9 is the lessons list if you read nothing
else.

## "Is the Q4 quantization bad?"

Q4 (group-64 int4) has ~11% rms weight error, but per-dot output error is
~0.005-0.013 (error cancellation over 2048-dim dots). It's the *router* that
amplifies it — a 1% input error can flip which experts are chosen, and the
hypersensitive last layer turns that into long-gen repetition (§7.3). So: Q4 is
fine for short answers, bad for long generations, and the fix (bf16-exact
tier) exists where the model fits.

## "Why are the speed numbers different across the docs?"

Because they're different things measured at different stages:
- **~4.9-10 tok/s** — earlier kernel generations (kernel3, batch B=4-6).
- **~7-10 tok/s** — the final single-stream floor, measured (Entry 50).
- **19.6 tok/s @ B=8** — standalone kernel benchmark with dummy tokens whose
  routing collides (best case, Entry 48).
- **~3-5 tok/s** — real serve aggregate with real prompts (Entry 70).

The paper's appendix (§11) labels each one; USAGE §8 matches.

## "Why did you use Python for the router but C for the kernel?"

The design principle was "simple > complex" (ARCHITECTURE §8): Python for
everything that isn't the hot loop (control plane, routing, session
management, verify), C only where the per-token cost dominates (attention,
experts, lm_head). The one C call per token boundary kept the Python flexible
and the C fast — and every arch flag is registry-driven, so Python changes
don't touch C.

## "How do I know the kernel is correct?"

Two independent gates, both automated:
1. **`verify.py --native <model>`** — native prefill logits + greedy vs HF
   transformers on 2 chat queries; PASS = argmax match + greedy-identical
   (margin-aware near-tie acceptance). docs/ARCH_QA_AUDIT.md documents how a
   silently-wrong QK-norm flag passed coherence checks but failed this.
2. **`batch_correctness_test.py`** — batch vs single-stream: B=1 bit-identical,
   B=2-4 ≤ 7e-7 (fp32 rounding).

Plus every component (attention, QK-norm, RoPE, routing, expert FFN, KV) was
individually compared against HF to corr ≥ 0.999 (§3.2).

## "What does 'verified on 5 models' actually mean?"

Native-vs-HF PASS on Llama-3.2-1B, Qwen2.5-1.5B, SmolLM2-1.7B, Qwen3-8B,
OLMoE-1B-7B (Entry 67) — each: greedy-identical on 2 short chat queries, with
max|dlogit| 0.16-2.88. Note the two caveats: (1) it's short-horizon (≤16
tokens) — Q4 long-gen drift is a separate, known issue; (2) the 30B itself was
never HF-verified because its bf16 reference can't load on this machine.

## "Is this a thesis / paper?"

It's written like a paper (and the retrospective could be worked into a
workshop submission — the negative results and the router-sensitivity finding
are the publishable parts). As-is it's the engineering record + lessons, MIT
licensed for anyone to build on.

## "What surprised you most?"

That the failures were more informative than the successes — and that the
biggest surprise was a hardware division problem: 4 GB/token ÷ 35 GB/s ≈ 8
tok/s. Everything "smart" (layer skip, expert cache, spec decode, masking)
lost to a number you can compute on a napkin before writing any code (§9.1).

## "What did building this teach you that reading didn't?"

First-hand understanding of the machinery — the kind that papers and videos
don't convey:

- **Tokens**: at the boundary, everything is token IDs; the tokenizer defines
  the model's world.
- **Layers are sequential state**: each layer reads and writes the same
  residual vector; attention and FFN are deltas. Skipping a layer breaks the
  KV-coupled state downstream — intuition you only get by trying it and
  watching the output collapse.
- **Depth behaves differently per family**: a 36-layer dense model has a
  skip-safe tail; a 16-layer MoE has none. "Redundancy scales with depth" is a
  measured fact here, not a slogan.
- **Architectures are not one thing**: qwen2 vs qwen3 vs olmoe vs mixtral
  differ in QK-norm placement, RoPE scaling, bias, router renormalization, and
  expert naming — the registry is the full catalog (§7.2). Marketing says
  "one architecture"; building says otherwise.
- **The last layer is where power and fragility hide**: OLMoE L15 amplifies
  6 → 97 in one layer; a small upstream error becomes a huge downstream flip.
- **Hardware floors are unforgiving**: 4 GB/token ÷ 35 GB/s is not an opinion,
  it's a division problem.

## "What would you do differently?"

1. **Measure the bandwidth floor first** — before writing any kernel
   (JOULE_PAPER §9.1).
2. **Check the literature harder on inference-time skipping** — the
   re-derivations cost weeks (the measurements are still useful, the outcomes
   were predictable).
3. **Get GPU access earlier** — the paradigm's best case is on an accelerator.
4. **Verify long generations from day one** — the ≤12-token verify horizon hid
   the drift for two sessions (Entries 71-73).
5. **Build the compare-against-HF harness before the kernel** — it resolved
   every dispute (§9.11).
