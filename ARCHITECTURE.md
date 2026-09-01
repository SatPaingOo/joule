# Joule — Architecture: Database-Style AI Inference

> **Design principle:** Store weights on disk. Query on demand. Load only
> what's needed. Release after use. Scale to any model size.

> **STATUS (2026-08-31)**: This document describes the *aspirational* design.
> What is actually built and verified: a native C kernel + arch registry +
> control plane + session-managed serve (see [docs/JOULE_PAPER.md](docs/JOULE_PAPER.md)
> and [docs/USAGE.md](docs/USAGE.md)). The "Sense Router" (§3), cloud
> escalation, and query-adaptive depth (§4.3) are **not implemented** — and
> per-query layer selection was **disproven** by measurement (Entry 43; see
> the honest correction in §4.1). Read this as the original vision; the built
> system is documented in docs/.

---


## 1. System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                            │
│         CLI / VS Code / API / Dashboard                      │
└────────────────────────┬────────────────────────────────────┘
                         │ OpenAI-compatible API
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              SENSE ROUTER (Python, ~10MB)                    │
│                                                              │
│  Query arrives → S(x) computed → routing decision:          │
│                                                              │
│  S(x) ≥ τ_cache → CACHE HIT (serve from answer store)       │
│  S(x) ≥ τ_local → LOCAL MODEL (selective load → compute)    │
│  S(x) < τ_local → CLOUD (optional escalation)               │
│                                                              │
│  Sense signals:                                              │
│    A(x) = cache alignment (embedding similarity)            │
│    C(x) = query consensus (neighborhood clustering)         │
│    M(x) = model readiness (probe confidence)                │
│                                                              │
│  Online calibration: (S, outcome) pairs → τ adjustment      │
└──────────────────────┬────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐ ┌──────────┐ ┌──────────────┐
   │ ANSWER     │ │ LOCAL    │ │ CLOUD        │
   │ CACHE      │ │ MODEL    │ │ PROVIDER     │
   │ (verified  │ │ (selective│ │ (BYOK or     │
   │  answers)  │ │  loading) │ │  managed)    │
   └────────────┘ └──────────┘ └──────────────┘
```

---

## 2. Weight Store — Database-Style Model Storage

### 2.1 Storage Format

```
model_store/
  ├── index.db              — weight index (layer → offset, size)
  ├── weights/
  │   ├── layer_001.bin     — layer 1 weights (SSD-optimized layout)
  │   ├── layer_002.bin     — layer 2 weights
  │   ├── ...
  │   └── layer_028.bin     — layer 28 weights
  ├── metadata.json         — model config, tokenizer, architecture
  └── sense_profile.json    — layer importance, exit points, thresholds
```

### 2.2 Storage Principles

| Principle | Implementation |
|---|---|
| **mmap** | Weights memory-mapped — OS pages in/out as needed |
| **SSD layout** | Weights stored contiguously per layer — sequential reads |
| **Index** | Layer → (file, offset, size) — O(1) lookup |
| **Selective load** | Only mmap the layers needed for this query |
| **Release** | munmap after computation — RAM returns to OS |
| **Hot weights** | Frequently used layers cached in RAM (LRU) |
| **Cold weights** | Rarely used layers stay on disk |

### 2.3 Weight Index

```json
{
  "layers": {
    "1": {"file": "weights/layer_001.bin", "offset": 0, "size": 52428800},
    "2": {"file": "weights/layer_002.bin", "offset": 0, "size": 52428800},
    "..."
  },
  "attention_weights": {...},
  "ffn_weights": {...},
  "lm_head": {"file": "weights/lm_head.bin", ...}
}
```

### 2.4 Selective Loading

```python
# Only load layers needed for this query
needed_layers = sense_router.get_required_layers(query)

# mmap only those layer files (OS handles paging)
for layer_idx in needed_layers:
    weights = mmap(layer_file[layer_idx])
    compute_with(weights)

# Release after computation
del weights  # OS reclaims pages
```

---

## 3. Sense Router — Query Analysis & Routing

### 3.1 Sense Signals

| Signal | Formula | Purpose | Cost |
|---|---|---|---|
| A(x) | `cos(emb(x), emb(key_i))` | cache alignment | ~1ms (embedder) |
| margin | `s1 − s2` | match confidence | ~0ms |
| C(x) | `skew(s₁..s_k)` | neighborhood stability | ~0ms |
| M(x) | `cos(h₁, h₂)` from 2-layer probe | model readiness | ~0.5s |

### 3.2 Routing Decision

```python
def route(query, config):
    s = sense_score(query)
    
    if s.cache_hit_confident:          # verified answer available
        return SERVE_FROM_CACHE        # ~0.5s, free
    
    elif s.local_model_sufficient:     # easy query
        return SERVE_LOCAL(small_model) # ~2-5s, cheap
    
    elif s.cloud_escalation_needed:    # hard query
        return SERVE_CLOUD(large_model) # ~10-30s, expensive
    
    else:
        return SERVE_LOCAL(default)     # fallback
```

### 3.3 Online Self-Calibration

```python
# Every serve records (S, outcome) pair
calibration_log.append((sense_score, user_satisfied))

# Every N queries → recalibrate thresholds
if len(calibration_log) >= 100:
    tau_s = optimize_threshold(calibration_log, target_satisfaction=0.9)
```

---

## 4. Selective Computation Engine

### 4.1 Granular Layer Loading

> **Honest correction (2026-08-31, measured in `layer_skip_probe.py` on
> Qwen3-8B):** whole-layer *skipping* is mathematically possible only via the
> residual stream (`h_{l+1} = h_l + f_l(h_l)` — skip `f_l`, keep `h`), but
> layer influence is **model-inherent, not query-dependent**: only ~4/36
> layers are skip-safe (single-layer-skip logit drift < 0.5), and the same
> layers are low-influence for every input. So per-query layer selection does
> NOT exist; selectivity lives at **neuron/expert granularity** (FFN ~50%
> inactive per token) and **MoE router granularity** (the model's own
> top-k). See `results/VALIDATION_LOG.md` Entry 43.

```python
@contextmanager
def load_layers(backend, layer_indices: list[int]):
    """Context manager: load only specified layers, release after.

    NOTE: for transformer models the full layer chain is sequential — every
    layer's output feeds the next. The real selective knobs are:
      - MoE: only the top-k experts per token (model-defined, exact)
      - Dense: only the top-mass FFN neurons (approximate, verify-gated)
    """
    loaded = backend.load_layers(layer_indices)
    try:
        yield loaded
    finally:
        backend.release_layers(layer_indices)
```

### 4.2 Layer Importance Profile

```python
def compute_layer_importance(model, calibration_queries):
    """Training-free: Block Influence + saturation analysis.
    Returns: importance score per layer (higher = more important)."""
    for layer_idx in range(model.num_layers):
        # Block Influence: cosine between layer in/out
        bi = 1 - avg_cosine(layer_input, layer_output)
        
        # Saturation: how often can we exit at this layer?
        sat = avg_cosine(h_layer_k, h_layer_k_minus_1)
        
        importance[layer_idx] = {
            "block_influence": bi,
            "saturation": sat,
            "skip_safe": sat > threshold,
        }
    
    return importance
```

### 4.3 Query-Adaptive Depth

```python
def adaptive_depth(query, layer_importance, config):
    """How many layers does THIS query need?
    Based on: query complexity + layer saturation + confidence."""
    
    # Run shallow probe (first k layers)
    probe_output = forward(query, layers=range(config.probe_depth))
    
    # Check confidence at probe depth
    confidence = logit_margin(probe_output)
    
    if confidence >= config.confidence_threshold:
        return config.probe_depth          # exit early
    else:
        return model.num_layers            # need all layers
```

---

## 5. Answer Cache — Persistent Knowledge Base

### 5.1 Cache Structure

```
answer_cache/
  ├── index.db              — query embedding → answer mapping
  ├── entries/
  │   ├── entry_0001.json   — {question, answer, verified, timestamp}
  │   ├── entry_0002.json
  │   └── ...
  └── embeddings.bin        — question embeddings for similarity search
```

### 5.2 Cache Operations

| Operation | When | Cost |
|---|---|---|
| **Populate** | After FULL decode | 1 embed computation |
| **Lookup** | Before FULL decode | 1 embed + cosine search (~1ms) |
| **Verify** | After cache hit | 1 teacher-forced pass |
| **Evict** | LRU when capacity exceeded | 0 (just delete) |

### 5.3 Cache Growth

```
Day 1:   0 entries  (cold start)
Day 7:   ~50 entries (user's common questions cached)
Day 30:  ~200 entries (most support queries covered)
Month 3: ~500+ entries (80%+ hit rate for regular users)

= Gets better with use. No conversion needed. No training needed.
```

---

## 6. Implementation Modules

```
src/jouleai/
  __init__.py
  config.py               # SensePointConfig
  interfaces.py           # Protocols (IBackend, ICacheStore, IVerifier)
  
  storage/                # Weight store (database-style model storage)
    __init__.py
    index.py              # weight index (layer → file, offset)
    store.py              # weight storage (mmap, SSD layout)
    loader.py             # selective loader (context manager)
  
  compute/                # Selective computation
    __init__.py
    selective.py          # forward with only loaded layers
    depth.py              # query-adaptive depth
    release.py            # release weights after computation
  
  sense/                  # Sense routing
    __init__.py
    signals.py            # A(x), C(x), M(x) computation
    router.py             # routing decision (cache/local/cloud)
    calibration.py        # online self-calibration
  
  cache/                  # Answer cache
    __init__.py
    store.py              # answer store (persistent)
    embeddings.py         # query embedding for similarity
    verify.py             # verification logic
  
  backends/               # Inference backends
    __init__.py
    hf_backend.py         # HuggingFace backend (prototype)
    llamacpp_backend.py   # llama.cpp backend (production)
  
  serve/                  # Serving layer
    __init__.py
    server.py             # FastAPI OpenAI-compatible server
    cli.py                # CLI interface
```

---

## 7. Implementation Priority

| Phase | Component | Why First |
|---|---|---|
| **P1** | Weight store + selective loader | Core paradigm — proves the concept |
| **P2** | Sense router (train-free signals) | Routing without training data |
| **P3** | Answer cache + verify | Lossless guarantee |
| **P4** | llama.cpp backend | Production inference (C++ speed) |
| **P5** | OpenAI-compatible server | User-facing API |
| **P6** | Query-adaptive depth | Advanced optimization |

---

## 8. Design Principles

1. **Storage ≠ RAM** — weights live on disk, RAM is workspace only
2. **Selective > Complete** — load only what's needed per query
3. **Release > Retain** — return RAM after computation
4. **Verify > Trust** — every cached answer is verified before serving
5. **Learn > Configure** — system calibrates from actual usage
6. **Simple > Complex** — minimal dependencies, fast startup
