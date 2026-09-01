# Joule

> **A negative-results engineering retrospective** — serving a 61 GB MoE
> (Qwen3-30B-A3B) on a 31 GB laptop: what worked, what failed, and the
> hardware floor no software can escape.

Joule is an inference engine that stores model weights on disk and loads only
the active set per token, so a model **larger than RAM** can be served. That
part works — this repo is the complete engineering record of it, including the
parts that didn't.

**Proven (measured, this repo):**
- A 61 GB MoE served on 31 GB RAM, RAM ∝ working set, outputs budget-invariant
- A registry-driven, arch-generic native C kernel verified against HF on 5 models
- Batch amortization is the only CPU speed lever for large models

**Disproven (by this project's own measurements):**
- Per-query layer selection — layer influence is model-inherent, not
  query-dependent (Entry 43)
- "30-150 tok/s on a laptop" — single-stream decode is memory-bandwidth-bound
  (~7-10 tok/s floor); the measured real serve aggregate is ~3-5 tok/s
- Inference-only layer skipping, expert-output caching, cross-family spec
  decoding

Every claim here traces to
[results/VALIDATION_LOG.md](results/VALIDATION_LOG.md) (Entries 16-73).

## The Original Vision (and what survives)

The idea that launched the project: LLM inference should work like a database —
weights on disk, load only what a query needs, release after use.

```
Traditional (Ollama, vLLM):
  Model Load → ALL weights into RAM (4-15 GB) → stays there forever
  RAM usage = model size (constant, regardless of query)

Joule (Database-Style):
  Weights stored on disk → query arrives → selective load → compute → release
  RAM usage ∝ what's needed for THIS query (granular, adaptive)
```

| | Traditional | Joule (as envisioned) |
|---|---|---|
| Model storage | RAM (always loaded) | **Disk (loaded on demand)** |
| RAM usage | Model size (constant) | **∝ active set (MoE) — measured** |
| Loading | All-at-once | **Selective per query** |
| After compute | Weights stay in RAM | **Released back to disk — measured** |
| 1TB model | 1TB RAM needed | **served if the active set fits (MoE)** |

**What survives measurement**: the *storage* half — weights on disk, the MoE
router's active set loaded per token, release after use. RAM ∝ working set is
real.

**What failed**: the *per-query layer selection* half ("this query needs layers
1,2,3,25") — layer influence is model-inherent, not query-dependent. See the
status note in [VISION.md](VISION.md) and [docs/JOULE_PAPER.md](docs/JOULE_PAPER.md) §6.

## Key Principles

1. **Load only what you need** — selective weight loading per query
2. **Compute only what you use** — skip redundant layers/blocks
3. **Release after use** — RAM returns to baseline after computation
4. **Gets better with use** — cache accumulates, sense layer learns
5. **Verify-gated** — the verify gate checks greedy identity vs HF on short
   queries (not logit-exact); long generations can drift under Q4
   quantization (see docs/JOULE_PAPER.md §6.1)

## Status

🔬 **v0.4 — Working prototype.** Config-driven engine (7 arch families),
native C batch-decode kernel (ggml-exact spin barrier, Q4 batch GEMM,
correctness bit-identical vs single-stream), OpenAI-compatible server
serving Qwen3-30B-A3B on a 31 GB laptop. Honest physics: single-stream
decode is memory-bandwidth-bound (~7-10 tok/s floor); the measured batch
aggregate is ~3-5 tok/s real serve (kernel best case 19.6 tok/s @ B=8 with
dummy tokens) — see docs/JOULE_PAPER.md for the full numbers.

**Platform**: the native path (the committed C DLLs) is **Windows x64**;
the pure-Python path works on any OS but is the slow prototype path.
CPU-only — no GPU required.

## Getting Started

**Models are NOT committed** (they're large, user-downloaded). To run Joule:

```bash
# 1. Install deps (CPU-only; the C DLLs are committed, no compiler needed)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .        # installs the rest + `joule-convert` / `joule-serve` scripts

# 2. Download a verified MoE model (serve is MoE-only; OLMoE is the smallest
#    verified one — ~13 GB, no HF token needed)
huggingface-cli download allenai/OLMoE-1B-7B-0824-Instruct --local-dir models/OLMoE-1B-7B-0824-Instruct

# 3. Convert (builds the Q4 store) + verify vs HF
joule-convert models/OLMoE-1B-7B-0824-Instruct --verify

# 4. Serve — browser chat at http://127.0.0.1:8080/chat
joule-serve models/OLMoE-1B-7B-0824-Instruct --backend native
```

(`PYTHONPATH=src python -m jouleai.cli.joule_* ...` also still works — the
launcher and USAGE.md use that form.)

**Serve currently supports MoE models only** (the kernel's dense path is
verified but not wired into serve — see [docs/USAGE.md](docs/USAGE.md) §8).

Verified models (native-vs-HF PASS, Entry 67): Llama-3.2-1B, Qwen2.5-1.5B,
SmolLM2-1.7B, Qwen3-8B, OLMoE-1B-7B. The flagship demo is Qwen3-30B-A3B
(61 GB bf16 → 15.4 GB Q4) served on a 31 GB laptop with RAM ∝ working set
(Entry 51) and budget-invariant outputs (Entry 22). Long generations under
Q4 quantization drift into repetition — the bf16-exact tier fixes it
(Entry 73). Every claim here traces to an Entry in
[results/VALIDATION_LOG.md](results/VALIDATION_LOG.md) (now committed).
Full details: [docs/USAGE.md](docs/USAGE.md).

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
- [VISION.md](VISION.md) — the granular computing paradigm
- [ROADMAP.md](ROADMAP.md) — implementation plan
- [docs/USAGE.md](docs/USAGE.md) — run / API / clients
- [docs/BATCH_DECODER.md](docs/BATCH_DECODER.md) — batch kernel design + measured progress
- [docs/JOULE_PAPER.md](docs/JOULE_PAPER.md) — **the retrospective**: what worked, what failed, the hardware floor
- [docs/JOURNEY.md](docs/JOURNEY.md) — what was proven, in order
- [docs/FAQ.md](docs/FAQ.md) — anticipated questions & honest answers
  ("isn't this known research?", "why CPU-only?", "is it lossless?", ...)
- [results/VALIDATION_LOG.md](results/VALIDATION_LOG.md) — experiment log
  (Entries 16-73, committed — the evidence base for every claim here)

## License

[MIT](LICENSE) — free to use, modify, and distribute (see LICENSE for the full terms).
