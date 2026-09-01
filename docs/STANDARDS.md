# Joule — Coding Standards & Project Structure
> **Context**: code standards (SOLID/OOP). See [docs/EDITORS.md](EDITORS.md)
> (dev setup), [docs/USAGE.md](USAGE.md) (run).

---

> 2026-08-30 | The project is standard, structured, SOLID/OOP-conformant.
> Every new module must follow these rules.

## 1. Directory Structure (standard)

```
src/jouleai/
├── arch/          → registry.py (ArchSpec, config-driven; 11 archs incl.
│                     deepseek MLA, gemma, mixtral), verify.py (auto-PASS harness)
├── control/       → control plane (the single place that decides HOW a model runs):
│   ├── device.py  → cross-OS detection (RAM/CPU/GPU/NPU/bandwidth, Windows/macOS/Linux)
│   ├── selector.py→ device+model → ExecutionPlan (quant/backend/batch/spec/sparsity/budget)
│   └── controls.py→ ControlCenter (one control point: device + plan + live health)
├── cli/           → joule_convert.py, joule_serve.py  (entry points)
├── engine/        → generic_streamer.py (flag-driven forward + forward_batch),
│                     stream_engine.py, masked_mlp.py
├── governor/      → resource_governor.py (auto+manual device control)
├── monitor/       → resource_monitor.py (RAM/CPU/power sampling)
├── native/        → C kernels + OOP backends + build script
│   ├── quant_gemv.c/.dll    → Q4-dequant GEMV (scalar)
│   ├── expert_ffn.c/.dll    → AVX2 fused expert + Q4 GEMM
│   ├── decode_kernel.c/.dll → full decode kernel (single + batch, spin pool)
│   ├── decoder3.py          → NativeDecoder (ctypes harness, batch KV)
│   ├── batch_bench.py       → batch throughput benchmark (B=1..16)
│   ├── batch_correctness_test.py → batch-vs-single correctness gate
│   ├── backend.py           → KernelBackend (abstract + AVX2/Scalar/Decode)
│   ├── build_native.py      → one-command rebuild
│   └── kernel.py, moe.py    → thin ctypes adapters
├── routing/       → mask_policy.py, probe_bank.py
├── session/       → session_manager.py (multi-chat, batch scheduler)
├── storage/       → weight_store.py (mmap rows), expert_store.py (LRU pool),
│                     q4_store.py (Q4 records)
└── experiments/   → proto_*.py (phase A-F + spec/batch tests) — RESEARCH
                        scripts; **production code must never import from here**
```

**Hard rules (enforced by convention):**
- All kernel DLLs load via `jouleai.native.backend.get_dll(name)` — the single
  access point (no `ctypes.CDLL("...dll")` scattered in callers).
- `src/jouleai/experiments/` is the research record — one-off probes and phase
  prototypes. Production (`cli/`, `engine/`, `native/decoder3.py`, `serve/`)
  must not import from it. Promote a working prototype into `engine/` or
  `native/` before wiring it into serve.

## 2. Naming Conventions

| Thing | Rule | Example |
|---|---|---|
| Files | snake_case, semantic (what it does, not v1/v2) | `quant_gemv.c`, `expert_ffn.c` |
| Classes | PascalCase | `KernelBackend`, `SessionManager` |
| Functions/methods | snake_case | `forward_batch`, `q4_gemm` |
| Constants | UPPER_SNAKE | `MAX_CONCURRENT` |
| DLLs | match source name | `quant_gemv.c` → `quant_gemv.dll` |
| No version suffixes | `kernel1/2/3` is banned | semantic names only |

## 3. SOLID Mapping

| Principle | Where applied |
|---|---|
| **S** (single responsibility) | One class = one job: `SessionManager` owns sessions, `ResourceGovernor` owns device control, `KernelBackend` owns compute dispatch |
| **O** (open/closed) | New ISA (AVX-512, NPU) = new `KernelBackend` subclass — no caller changes |
| **L** (Liskov) | `ScalarBackend` / `AVX2Backend` / `DecodeBackend` all satisfy the same interface — substitutable |
| **I** (interface segregation) | Backends expose only `q4_gemv/q4_gemm`; the engine never touches concrete DLLs |
| **D** (dependency inversion) | Engine depends on `KernelBackend` (abstract), not on `ctypes.CDLL("...")` — DLLs are injected |

## 4. OOP Patterns

- **Abstract base + implementations**: `KernelBackend(ABC)` → `AVX2Backend`, `ScalarBackend`, `DecodeBackend`
- **Factory**: `KernelBackend.auto()` picks the best ISA for the machine
- **Strategy**: `MaskPolicy(ABC)` → `TopMassPolicy`, `ThresholdPolicy`
- **Composition over inheritance**: engine *has* a backend/pool/governor, not *is* one

## 5. Quality Gates (every merge)

1. **Verify harness PASS** for any arch change (`joule convert --verify`;
   `python src/jouleai/arch/verify.py <model> --native` for the native kernel)
2. **Correctness**: batched/token outputs identical to reference (Entry 36: 3/3;
   Entry 42: batch B=1 bit-identical, B=2-4 ≤ 7e-7 — run
   `python src/jouleai/native/batch_correctness_test.py`)
3. **No version-suffixed names** (kernel1/2/3, file_v2) — semantic only
4. **Build reproducibility**: `python src/jouleai/native/build_native.py` rebuilds all
5. **Docs updated**: VALIDATION_LOG entry + relevant doc (USAGE/JOURNEY/STANDARDS)
6. **Arch switch rule (Entry 67)**: the native kernel must NEVER be a
   per-model specialization — all arch flags come from the registry
   (`get_spec` → KernelCfg), the workspace is heap-sized from cfg (no static
   caps), and every new family passes `verify_native` before serve exposure.
   See docs/ARCH_QA_AUDIT.md for the audit + flag matrix.

## 6. Why this matters (the product promise)

Clean structure + SOLID/OOP = the "any model on any device, batch-aggregate"
stack stays maintainable as we add: new arch families (registry), new ISAs
(backend subclasses), new resources (governor), new concurrency (sessions).
Standard code is what lets other people read, review, and trust the project.
