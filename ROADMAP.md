# Joule — Implementation Roadmap

> Step-by-step plan from architecture design to working product.
> Each phase produces a testable artifact before moving to the next.

> **STATUS (2026-08-31)**: This roadmap predates the pivot to a native C
> kernel. What is actually built: a working product loop (`joule convert` →
> `joule serve` → OpenAI-compatible HTTP + browser chat) on a registry-driven
> arch-generic kernel — see [docs/USAGE.md](docs/USAGE.md). The phases below
> (llama.cpp backend, Sense Router) were **superseded** by the native C
> kernel (zig, nostdlib) and the control plane. Read as history, not current
> plan.

---

## Phase 1 — Weight Store Prototype (Week 1-2)

**Goal:** Prove selective weight loading works — load only needed layers.

**Deliverables:**
- [ ] `storage/store.py` — mmap weight storage (GGUF → layer-per-file)
- [ ] `storage/loader.py` — selective loader (context manager)
- [ ] `storage/index.py` — weight index (layer → offset mapping)
- [ ] Test: load Qwen2.5-1.5B layers 1-8 only → verify RAM usage < full load
- [ ] Test: compute with partial layers → output correctness

**Success criteria:** RAM usage ∝ loaded layers (not model size).

---

## Phase 2 — Sense Router Prototype (Week 2-3)

**Goal:** Route queries without training data.

**Deliverables:**
- [ ] `sense/signals.py` — A(x) embedding, C(x) consensus, M(x) probe
- [ ] `sense/router.py` — routing decision (cache/local/cloud)
- [ ] Calibration: collect (S, outcome) pairs from real usage
- [ ] Test: routing accuracy on 50-query workload

**Success criteria:** Router correctly routes ≥90% of queries (measured by verify gate).

---

## Phase 3 — Answer Cache (Week 3-4)

**Goal:** Persistent answer cache with lossless verification.

**Deliverables:**
- [ ] `cache/store.py` — persistent answer store (disk-backed)
- [ ] `cache/verify.py` — Tier-A verification (teacher-forced pass)
- [ ] Cache hit → serve (<1s, token-identical)
- [ ] Cache miss → FULL decode → cache new answer
- [ ] Test: 10-query repeat benchmark (expect 10x on repeats)

**Success criteria:** Cache hit <1s, verify pass rate ≥97%, token-identical outputs.

---

## Phase 4 — llama.cpp Backend (Week 4-6)

**Goal:** Replace PyTorch with llama.cpp for production inference.

**Deliverables:**
- [ ] `backends/llamacpp_backend.py` — llama.cpp HTTP client
- [ ] Remove PyTorch dependency for inference (keep for conversion only)
- [ ] Benchmark: joule-serve vs ollama-serve (same model, same prompts)
- [ ] Target: lighter install, faster startup, equal/better speed

**Success criteria:** Install <100MB (vs PyTorch 5GB), startup <5s, speed ≥ Ollama.

---

## Phase 5 — OpenAI-Compatible Server (Week 6-8)

**Goal:** User-facing product — drop-in replacement for any OpenAI API client.

**Deliverables:**
- [ ] `serve/server.py` — FastAPI server (chat completions, streaming, health)
- [ ] `serve/cli.py` — CLI chat interface
- [ ] `jouleai-convert` — model analyzer (sense profile generator)
- [ ] Dashboard: usage stats, cache hit rate, resource usage
- [ ] Documentation: README, API docs, deployment guide

**Success criteria:** A new user can install → serve → chat in <5 minutes.

---

## Phase 6 — Advanced Features (Week 8+)

- [ ] Query-adaptive depth (CALM-style early exit)
- [ ] MoE support (DeepSeek-V2-Lite dynamic-k)
- [ ] Online self-calibration (auto τ_s from production traffic)
- [ ] Multi-model routing (1.5B for easy, 7B for hard)
- [ ] VS Code extension
- [ ] Paper submission (ArXiv preprint)

---

## Deferred (from BACKLOG)

| Item | Trigger |
|---|---|
| Cache auto-persist | After Phase 3 |
| fp32 verify | After Phase 4 benchmark |
| Vector DB upgrade | When cache >10k entries |
| 200+ calibration | Paper phase |
| GPU validation | When GPU available |
| Tier B formal proof | Paper phase |
