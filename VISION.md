# Joule — Vision: Database-Style AI Inference

> **"AI models should work like databases — store everything, query only what you need."**
> **STATUS (2026-08-31)**: This is the *vision* — the long-term paradigm.
> The built, measured reality is in [docs/JOULE_PAPER.md](docs/JOULE_PAPER.md)
> and [docs/USAGE.md](docs/USAGE.md). Two vision claims were **disproven** by
> this project's own measurements: per-query layer selection ("this query
> needs layers 1,2,3,25...") — layer influence is model-inherent, not
> query-dependent (Entry 43); and "1TB models on 8GB RAM at speed" — decode is
> memory-bandwidth-bound (see paper §4). What DID work: MoE expert streaming
> (RAM ∝ working set, budget-invariant) and a verified native kernel.

---

## The Problem

Every LLM inference engine today (Ollama, vLLM, TGI) follows the same pattern:

```
1. Load ENTIRE model into RAM (4-15 GB)
2. Keep it there permanently
3. Every query reads ALL weights (regardless of relevance)
4. RAM usage = model size (constant)
```

This works for servers with 80GB GPUs. It fails for:
- Laptops with 8-16GB RAM
- Edge devices with 4-8GB RAM
- Battery-powered devices where every computation costs power
- 1TB+ models that can't fit in any consumer RAM

## The Insight

**A model is a KNOWLEDGE BASE, not a monolithic blob.**

Just like a database:
- Stores data on disk
- Loads only the relevant pages for each query
- Releases memory after the query completes
- Scales to any size without loading everything

A model's weights can be stored and queried the same way:
- Store weights on disk (SSD-optimized format)
- Load only the relevant layers/weights for each query
- Release after computation
- Scale to ANY model size without loading everything

## The Paradigm

```
┌─────────────────────────────────────────────────┐
│              WEIGHT STORE (Disk/SSD)              │
│                                                   │
│  Layer 1  [████████████████████████████]  on disk │
│  Layer 2  [████████████████████████████]  on disk │
│  ...                                              │
│  Layer 28 [████████████████████████████]  on disk │
│                                                   │
│  + Index: "query type → weight location"         │
└──────────────────────┬──────────────────────────┘
                       │
                       │ Query arrives
                       │ Sense layer decides:
                       │   "This query needs layers 1,2,3,25,26,27,28"
                       │
                       ▼ Selective load
┌─────────────────────────────────────────────────┐
│              RAM (only what's needed)             │
│                                                   │
│  Layer 1  [████████████]  loaded                  │
│  Layer 2  [████████████]  loaded                  │
│  Layer 25 [████████████]  loaded                  │
│  Layer 26 [████████████]  loaded                  │
│                                                   │
│  = 7/28 layers = 25% of model = 3.75 GB          │
│  (vs 15 GB if all loaded)                         │
│                                                   │
│  → Compute → Release → RAM back to baseline      │
└─────────────────────────────────────────────────┘
```

## The Human Brain Analogy

The human brain has 86 billion neurons but:
- "How are you?" activates ~1% of neurons (instant response)
- "Quantum physics" activates ~5% of neurons (deeper processing)
- The rest remain at rest (zero power consumption)
- Knowledge is STORED in synaptic connections (like disk)
- Only RELEVANT neurons fire (like selective loading)
- After thinking, neurons return to rest (like releasing RAM)

**The transformer violates every one of these principles:**
- 100% activation for every query
- No rest state
- No selective loading
- No release after use

## The Vision

An inference engine where:
1. **Models are databases** — stored on disk, queried on demand
2. **Loading is selective** — only relevant weights per query
3. **RAM is bounded** — proportional to what's needed, not model size
4. **Power ∝ computation** — not model size
5. **ANY model size works** — 1TB models on 8GB RAM devices
6. **Gets better with use** — cache accumulates, patterns learned

## The Impact

| User | Benefit |
|---|---|
| Laptop user | Run 70B models on 16GB RAM |
| Edge device | Run 7B models on 4GB RAM |
| Enterprise | Run 1TB models without 1TB GPU |
| Battery user | Power ∝ computation, not model size |
| Privacy user | Everything on your device, nothing sent to cloud |
| Developer | One API, any model, any size |
