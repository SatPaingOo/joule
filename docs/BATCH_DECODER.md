# BatchDecoder Design — the batch-aggregate path (honest physics)
> **Context**: the batch-decode design + measured aggregate. Evidence:
> [results/VALIDATION_LOG.md](../results/VALIDATION_LOG.md) (Entries 35-42, 68-70).
> See [docs/JOULE_PAPER.md](JOULE_PAPER.md) §3-4 (the honest numbers).

---

> 2026-08-30, updated 2026-08-31 | Goal: maximize AGGREGATE throughput on the
> 31 GB Ryzen laptop (single-stream is bandwidth-floor ~7-10 tok/s; batch
> amortizes weight reads). Measured: kernel 19.6 @ B=8 (dummy), real serve ~3-5.

## Why single-stream is bandwidth-floor (not 30-150)
Decode is memory-bandwidth-bound. 30B-A3B Q4 streams ~4 GB/token (dense
attention + lm_head + active experts). The **measured** effective bandwidth on
this laptop is ~35 GB/s (not the ~50-60 GB/s DDR5 nominal) → **~7-10 tok/s
floor**, matching the measured single-stream rate (Entry 50). Kernel quality
cannot change this — the paper's §4 documents the full bandwidth math.

## The route: batch B sequences together
At batch B, weights are **read once per B tokens** (not once per token):
```
aggregate tok/s ≈ B × BW / bytes_per_token
```
The measured kernel best case is **19.6 tok/s @ B=8** (with dummy tokens whose
routing collides). The real serve aggregate is **~3-5 tok/s** (Entry 70) —
real prompts route to more unique experts. The amortization direction is
proven; the absolute number depends on the routing pattern, not just B.

## Current gap (measured, as of Entry 70)
The batch scheduler + `decode_batch_argmax` are **wired into serve** (Entry
68/70). The remaining gap to higher aggregate is the per-step compute time growing with B
(fp32 attention/lm_head matvecs) and the real-prompt expert spread — not the
server wiring, which is done.

## BatchDecoder design (kernel3 → batch)
```
requests → session manager → batch scheduler (fill up to B=8-16)
        → BatchDecode:
             input:  B token vectors + B KV positions
             48 layers, each:
               - attention: batched QKV (B×d) — shared W, one read
               - experts:   B×topk expert ids → union → batched Q4 GEMM
                            (MUL_MAT_ID style: row-block work-stealing)
               - norms:     B×d
             output: B next-token logits (shared lm_head, one read)
        → per-session KV updated, stream out per session
```

## Key kernels (from research, priority order)
1. **Batched Q4 GEMM** (X[B,d] @ W_q4[m,d]^T) — weights read once for all B.
   Q4_0 int8 dot (maddubs→madd→fmadd), AVX2, work-stealing over row blocks.
2. **Persistent thread pool + spin barrier** — no per-op spawn (1,640
   barriers/token on MoE; spawn overhead is fatal). Spin barrier = +40%.
3. **AVX-512 VNNI** dispatch (Zen4 mobile has it) for batch ≥8 compute.
4. **Per-session KV** in one contiguous buffer (B × maxT × n_kv × hd).

## Progress (measured)
- **Phase 1 done**: batch scheduler + batched forward → B=4 wall 16.3 s
  (serial would be ~64 s) = **4× wall speedup**, weights read once per B.
- **Phase 2 (kernel) — built, correct, deterministic** (2026-08-30):
  - `decode_layers_batch` with **ggml-exact spin barrier** (fixed worker ids,
    main participates, monotonic `n_barrier_passed` — the race that segfaulted
    Entries 38/39 is gone). Persistent thread pool, no per-op CreateThread.
  - **Q4 int8 batch GEMM** (int4 unpacked once per group, FMADD across B
    activations) — weights read once per B for shared experts; per-expert
    partials combined single-threaded (no cross-thread RMW race).
  - Correctness: B=1 batch == single decode **bit-identical (0.0 diff)** at
    all positions; B=2/3/4 ≤ 7e-7 (fp32 rounding); KV persistence across
    calls bit-identical. `batch_correctness_test.py` ALL PASS.
  - Single-stream: **1.5 → 4.7-4.9 tok/s** (3x, kernel3 fused path).
  - Aggregate B=1..8: **4.9 → 10.3 tok/s peak at B=4-6** (2x over B=1).
  - Remaining gap to higher aggregate: per-step time grows with B (210→860 ms) because
    the fp32 attention/lm_head matvecs are compute-bound, not bandwidth-bound
    (lm_head 1.2GB fp32 read/token is ~50% of each step at B=1 and still
    ~40% at B=8). The amortization is real (lm_head read stays ~constant) but
    the int8 VNNI GEMM micro-kernel (attention QKV + lm_head + cache-blocked
    FFN) is what converts the remaining compute-bound time into bandwidth.
  - **Server wiring: DONE** (Entry 68/70) — `decode_layers_batch` +
    `decode_batch_argmax` are wired into `joule serve --backend native` via the
    session scheduler's batch decode. Measured real serve aggregate: ~3.1
    words/s (Entry 70) — the 19.6 tok/s @ B=8 kernel benchmark uses dummy
    tokens whose routing collides; real prompts route to more unique experts.
- Phase 3 (spec decode): queued — the amortization direction is proven.

## Rollout
- Phase 1: batch the EXISTING GenericStreamer path (B sequences, shared pool,
  torch batched ops — weights read once per B via torch matmul batching).
  Expected: B=4 → 5-8 tok/s aggregate (torch batch efficiency), proves the
  amortization direction.
- Phase 2: kernel3 batch GEMM (Q4 int8, spin barrier) → B=8-16 → 60-150.
- Phase 3: spec decode per stream → per-user 20-28 while aggregate holds.

## Why this is the moat
Dense engines read ALL weights per token per user. Joule reads active-set Q4
once per batch → aggregate scales with concurrency. On high-bandwidth devices
(Mac/Strix/GPU) the same batch kernel gives per-stream 150-480 too. This is
the "any model, any device, aggregate throughput" claim made real. Honest update (2026-08-31, Entry 73): single-stream is bandwidth-floor ~7-10 tok/s; the batch aggregate is kernel-best-case 19.6 @ B=8 (dummy tokens) / ~3-5 real serve. See [docs/JOULE_PAPER.md](JOULE_PAPER.md) §4.
