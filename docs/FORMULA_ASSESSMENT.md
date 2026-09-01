# Formula & Engine Assessment — where we are, what can be improved
> **Context**: the cost-model analysis. Evidence:
> [results/VALIDATION_LOG.md](../results/VALIDATION_LOG.md).
> See [docs/JOULE_PAPER.md](JOULE_PAPER.md) (the retrospective), [docs/USAGE.md](USAGE.md) (run).

---

> 2026-08-30 | Reviewed against the research sprint (kernel engineering + batching).

## 1. The master equation — still correct, now sharper

```
π*(x) = argmin E[T(π,x)]  s.t.  V(π,x) = PASS
```

The research confirmed the physics: for a single stream the decode cost is
memory-bandwidth-bound (≈7-10 tok/s floor for 30B-A3B Q4 on this laptop's
measured ~35 GB/s effective bandwidth), so the
equation's real lever is **aggregate** throughput. Updated cost model:

```
T_agg(π, batch B) = B × W_bytes / BW        (weights read once per B tokens)
per-stream T(π)   = W_bytes / BW            (~7-10 tok/s floor, kernel-independent)
```

**Implication**: the formula should minimize aggregate E[T] over concurrent
requests, not per-request T. The π* decision now includes a **batching
dimension**: route requests into decode batches to amortize weight reads.

## 2. What the engine already does right (validated)

| Component | Status | Proof |
|---|---|---|
| Load-what-you-need (Q4 pool, budget) | ✅ | RAM ÷4, IO ÷19-30 |
| Lossless (MoE router = free mask) | ✅ | token-identical 3/3 (30B) |
| Registry (7 arch families, path-only) | ✅ | llama/mistral/qwen3 exact |
| Resource governor (auto+manual) | ✅ | budget/threads/precision/backend/profile |
| Multi-chat control (sessions, concurrency) | ✅ | 2 parallel, 16.5s each (not 33s) |
| Verify harness (auto-PASS per arch) | ✅ | caught llama-rope + zeros-FFN |

## 3. The gap: kernel engineering to raise aggregate throughput

Research-grounded priority (evidence from llama.cpp ggml source + PRs):

1. **Multi-sequence batching** (the ONLY route to higher aggregate):
   weights read once per B tokens. Measured: kernel best case 19.6 tok/s @ B=8
   (dummy tokens); real serve aggregate ~3-5 tok/s (Entry 70).
   Engine change: batch-vectorized kernels (multiple sequences' tokens decoded
   together, per-sequence routing/expert ids, batch-aware attention).
   **✅ Built + correctness-verified (Entry 42)**: `decode_layers_batch`,
   B=1 == single bit-identical, B=2-4 ≤ 7e-7. Measured B=4-6 → ~10 tok/s
   aggregate; **server wiring DONE** (Entry 68/70); int8 GEMM remains for the
   full aggregate target.
2. **Persistent thread pool + custom spin barrier** (no OpenMP): +40% on MoE
   graphs (1,640 barriers/token — spawn overhead is fatal; keep threads alive).
   **✅ Done (Entry 42)**: ggml-exact barrier ported (fixed worker ids, main
   participates, monotonic n_barrier_passed), WaitOnAddress sleep fallback.
3. **Q4 int8 dot kernels** (maddubs→madd→fmadd): 3-5× vs fp32 at batch-1,
   more at batch ≥8. **✅ Q4 batch GEMM built** (int4 unpack once per group,
   FMADD across B); the fp32 attention/lm_head matvecs are the remaining
   compute-bound bottleneck → int8 VNNI GEMM is the next kernel.
4. **AVX-512 VNNI** (Zen4 mobile has it): 1.5-2.5× compute phases. Pending.
5. **Speculative decoding** (close-in-size same-family draft): 1.8-2.8× per-stream.
   Prototyped (Entry 27/54/69: cross-family fails, 0.6B same-family acceptance ~0 —
   the draft must be close in size, e.g. 1.7B/4B for a 30B target).
6. Batch-1 attention decode tuning: +10-15%. Pending.

## 4. Formula improvements worth making

| Change | Why |
|---|---|
| Add `B` (batch) to π* | aggregate throughput is the real objective |
| Batch-aware router | route requests into decode batches (similar experts co-locate) |
| `T(x)` split: prefill vs decode vs IO | already separate; make batch-aware |
| Per-session π* | each session's cache/mask reuse is independent; budget per session |
| Online batch scheduling | greedy: fill decode batch up to compute-bound as requests arrive |

## 5. Engine architecture next step (multi-chat + batching)

```
requests → SessionManager (per-session context)
        → π* router (cache-hit → serve; else join decode batch)
        → BatchDecoder: B tokens × 48 layers in one pass
             - persistent thread pool + spin barrier
             - Q4 int8 GEMM (batch-aware, expert-id routing)
             - per-session KV in one buffer
        → streaming out per session
```

This is the kernel3 evolution: from per-token to **per-batch** (B tokens), which
is where higher aggregate throughput lives.

## 6. Honest assessment

- **Single-stream**: ~7-10 tok/s floor on this laptop (physics). Spec decode
  (close-in-size draft) could reach ~15-20. "50-150 per user" is NOT reachable
  on this hardware class; the batch aggregate (kernel best case 19.6 tok/s @ B=8,
  real serve ~3-5 tok/s) is the honest ceiling here.
- **On high-bandwidth devices** (Strix Halo/Mac/GPU): 150-480 tok/s per stream
  because bandwidth scales. The same software stack, device-dependent.
- The moat is unchanged: active-set Q4 kernels + budget-invariant MoE
  streaming + arch-agnostic registry + resource governor. Dense engines cannot
  match the per-device envelope.
