# Joule

> **Database-Style AI Inference**
> Load only what you need. Compute only what you use. Release after use.

Joule reimagines LLM inference as a **database query system** — instead of
loading entire models into RAM, it stores weights on disk and selectively
loads only the parts needed for each query.

```
Traditional (Ollama, vLLM):
  Model Load → ALL weights into RAM (4-15 GB) → stays there forever
  RAM usage = model size (constant, regardless of query)

Joule (Database-Style):
  Weights stored on disk → query arrives → selective load → compute → release
  RAM usage ∝ what's needed for THIS query (granular, adaptive)
```

## The Paradigm Shift

| | Traditional | Joule |
|---|---|---|
| Model storage | RAM (always loaded) | **Disk (loaded on demand)** |
| RAM usage | Model size (constant) | **∝ needed weights only** |
| Loading | All-at-once | **Selective per query** |
| After compute | Weights stay in RAM | **Released back to disk** |
| 1TB model | 1TB RAM needed | **~1% loaded per query** |

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

## Getting Started

**Models are NOT committed** (they're large, user-downloaded). To run Joule:

```bash
# 1. Install deps (CPU-only; the C DLLs are committed, no compiler needed)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers huggingface_hub psutil numpy safetensors

# 2. Download a verified MoE model (serve is MoE-only; OLMoE is the smallest
#    verified one — ~13 GB, no HF token needed)
huggingface-cli download allenai/OLMoE-1B-7B-0824-Instruct --local-dir models/OLMoE-1B-7B-0824-Instruct

# 3. Convert (builds the Q4 store) + verify vs HF
PYTHONPATH=src python -m jouleai.cli.joule_convert models/OLMoE-1B-7B-0824-Instruct --verify

# 4. Serve — browser chat at http://127.0.0.1:8080/chat
PYTHONPATH=src python -m jouleai.cli.joule_serve models/OLMoE-1B-7B-0824-Instruct --backend native
```

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
- [results/VALIDATION_LOG.md](results/VALIDATION_LOG.md) — experiment log
  (Entries 16-73, committed — the evidence base for every claim here)

## License

[MIT](LICENSE) — free to use, modify, and distribute (see LICENSE for the full terms).
