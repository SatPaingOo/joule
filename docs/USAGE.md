# Joule — Usage Guide (Run • API • Clients)

> This guide covers everything that works today.
> All commands run from the repo root.
> Formatted for both Windows PowerShell and Git Bash.
> **Why the numbers are what they are**: [docs/JOULE_PAPER.md](JOULE_PAPER.md)
> (the retrospective). Evidence: [results/VALIDATION_LOG.md](../results/VALIDATION_LOG.md).

---

## 0. Requirements (one-time)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers huggingface_hub psutil numpy safetensors
# (Zig compiler is only needed to rebuild the C kernels — the DLLs are committed)
```

Python 3.11+ | CPU-only | no GPU required

---

## 1. Model selection — what is usable today

**Models are not committed** — you download them. Verified models
(native-vs-HF PASS, Entry 67):

| Model | Download (HF) | Convert status |
|---|---|---|
| **Qwen3-30B-A3B-Instruct-2507** | `huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir models/Qwen3-30B-A3B-Instruct-2507` | ✅ serve-ready (Q4 store) |
| Llama-3.2-1B-Instruct | `huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir models/Llama-3.2-1B-Instruct` | convert + verify PASS |
| Qwen2.5-1.5B-Instruct | `huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct` | convert + verify PASS |
| SmolLM2-1.7B-Instruct | `huggingface-cli download HuggingFaceTB/SmolLM2-1.7B-Instruct --local-dir models/SmolLM2-1.7B-Instruct` | convert + verify PASS |
| Qwen3-8B | `huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B` | convert + verify PASS |
| OLMoE-1B-7B-0824-Instruct | `huggingface-cli download allenai/OLMoE-1B-7B-0824-Instruct --local-dir models/OLMoE-1B-7B-0824-Instruct` | convert + verify PASS (dlogit 2.88/1.59, Entry 67) |

> **Note**: Llama models are gated on HF — run `huggingface-cli login` first.
> The Qwen/OLMoE models are open and need no token.

**The fastest way to try Joule (3 commands):**

> **Serve is MoE-only** — the quickstart must use a MoE model. OLMoE-1B-7B
> is the smallest verified MoE (~13 GB download, no HF token). Dense models
> (Llama/Qwen2.5/Qwen3-8B) are verified in the kernel but **not yet wired
> into serve** (see §8).

```bash
# 1. Download a verified MoE model (OLMoE-1B-7B — ~13 GB)
huggingface-cli download allenai/OLMoE-1B-7B-0824-Instruct --local-dir models/OLMoE-1B-7B-0824-Instruct

# 2. Convert (builds the Q4 store) + verify vs HF
PYTHONPATH=src python -m jouleai.cli.joule_convert models/OLMoE-1B-7B-0824-Instruct --verify

# 3. Serve — browser chat at http://127.0.0.1:8080/chat
PYTHONPATH=src python -m jouleai.cli.joule_serve models/OLMoE-1B-7B-0824-Instruct --backend native
```

Then open [http://127.0.0.1:8080/chat](http://127.0.0.1:8080/chat) in a browser
and type a question. The OpenAI-compatible API is at
`POST http://127.0.0.1:8080/v1/chat/completions` — works with any OpenAI client.

**For the flagship demo (30B on a laptop — 15.4 GB Q4, ~8 GB RAM budget):**

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir models/Qwen3-30B-A3B-Instruct-2507
PYTHONPATH=src python -m jouleai.cli.joule_convert models/Qwen3-30B-A3B-Instruct-2507 --budget-gb 8
PYTHONPATH=src python -m jouleai.cli.joule_serve models/Qwen3-30B-A3B-Instruct-2507 --port 8080 --budget-gb 8
```

---

## 2. Convert — prepare a model (MoE models)

```bash
# PowerShell / Git Bash (from repo root)
.\joule.ps1 convert models/Qwen3-30B-A3B-Instruct-2507 --budget-gb 8
# (Git Bash: PYTHONPATH=src python -m jouleai.cli.joule_convert ...)
```

**Result**:
- `Q4 expert store` (61 GB → 15.4 GB, on disk)
- `joule_manifest.json` — model info + RAM budget recommendation
- **Report card** — layers / experts / working set / quality statement

Options:
| Flag | Meaning |
|---|---|
| `--budget-gb 8` | RAM pool budget for decode (must exceed the working set — shown in the report card) |
| `--verify` | cross-check against HF transformers (logits + greedy) → PASS/FAIL in the report card |

(Convert is seconds if the Q4 store exists; first time writes ~15 GB.)

---

## 2.5 Adding a new model ("just point at the folder")

```bash
# 1. Download the model into models/xxx-30B
PYTHONPATH=src python -m jouleai.cli.joule_convert models/xxx-30B --budget-gb 8 --verify
PYTHONPATH=src python -m jouleai.cli.joule_serve models/xxx-30B --port 8080 --budget-gb 8
```

- **If the model_type is in the registry AND the kernel implements it**
  (qwen2 / qwen3 / llama / mistral / olmoe / qwen3_moe / mixtral — dense or
  standard-MoE): arch detection, QK-norm style, GQA, rope, expert config,
  tokenizer — all read from config.json automatically ✅ **no code changes
  needed**
- **Detected but NOT implemented by the kernel** (deepseek MLA, gemma, phi,
  gpt_oss): the engine raises a clear "unsupported architecture" error —
  these need kernel math support before they can run (registry detects them,
  the kernel doesn't implement them yet)
- Serving also works directly — if no Q4 store exists the server builds it at
  startup (first start is slower)
- `--verify` catches small per-model differences even for known archs —
  **if it says PASS, it's safe to use** (2-query greedy identity; see
  docs/JOULE_PAPER.md §6.1 for the long-generation caveat)
- **If the model_type is not in the registry** (e.g. deepseek_v3, qwen3_next):
  a clear "unsupported architecture" message appears — an adapter must be
  added first (no release until the verify harness passes)

---

## 3. Serve — start the API server

**PowerShell (recommended — launcher sets PYTHONPATH):**
```powershell
cd <repo-root>   # where you cloned Joule
.\joule.ps1 serve models/Qwen3-30B-A3B-Instruct-2507 --port 8080 --budget-gb 8
```

**Git Bash:**
```bash
PYTHONPATH=src python -m jouleai.cli.joule_serve models/Qwen3-30B-A3B-Instruct-2507 --port 8080 --budget-gb 8
```

**PowerShell raw syntax** (without the launcher):
```powershell
$env:PYTHONPATH = "src"
python -m jouleai.cli.joule_serve models/Qwen3-30B-A3B-Instruct-2507 --port 8080 --budget-gb 8
```

Wait for "engine ready ... joule serve listening" — then it's usable.
(The first query is slower while the pool warms up.)

---

## 4. API Endpoints

### 4.0 Control plane (auto-adaptive)

Serve now runs through the **control plane** (`src/jouleai/control/`) — one
place that detects the device (OS, RAM, CPU, GPU, NPU, memory bandwidth) and
auto-selects how the model runs:

```bash
PYTHONPATH=src python -m jouleai.cli.joule_serve models/Qwen3-30B-A3B-Instruct-2507
# [control] windows 31GB RAM (24GB free) | 8c/16t | BW~90GB/s | GPU=... | tier=mid
# [control] plan: backend=pool precision=q4 budget=1.9GB threads=8 batch=4 spec=on
```

The plan adapts per device + model: budget ∝ active set (RAM stays small for
MoE), backend = native (fits RAM) or pool (disk-backed), batch/threads by
cores+bandwidth, spec decoding on/off. Manual flags override auto:

| Flag | Overrides |
|---|---|
| `--budget-gb N` | RAM budget (else auto ∝ active set) |
| `--auto-budget` | cap at 40% of free RAM |
| `--threads N` | CPU threads |
| `--precision q4\|bf16` | expert tier — **q4** (small, ~11% error, default); **bf16** (exact, no quantization drift, 2× IO — for models that fit, e.g. OLMoE 6.5GB). bf16 eliminates long-gen repetition (64 tokens, reps=0 vs q4's 10, Entry 73) |
| `--backend auto\|native\|pool` | engine backend |
| `--max-concurrent N` | decode batch / concurrent users |

Live view: `GET /v1/control` returns device + model + plan + health.

### 4.1 `POST /v1/chat/completions` — main (OpenAI-compatible)

**Non-streaming:**
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}],\"max_tokens\":64}"
```

**Python (OpenAI SDK — most existing tools use this):**
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="joule-local")
r = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=64,
)
print(r.choices[0].message.content)
print("joule stats:", r.joule)   # tok/s, IO, cache_hit
```

**Streaming:**
```python
stream = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Tell me a short story."}],
    max_tokens=128,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

**Request fields:**
| Field | Meaning |
|---|---|
| `messages` | OpenAI-style chat messages |
| `max_tokens` | max tokens to generate (default 256) |
| `stream` | `true` → SSE streaming |

**`joule` stats in the response:** `tok_s` (mixed speed), `io_mb_per_tok` (SSD traffic), `cache_hit` (true → zero compute cost)

### 4.2 `GET /v1/models` — model list
```bash
curl http://127.0.0.1:8080/v1/models
```

### 4.3 `GET /status` — current state
```bash
curl http://127.0.0.1:8080/status
```
Shows: pool resident GB, process RSS, cache entries/hits, requests served, last tok/s

---

## 5. Connecting existing chat clients

OpenAI-compatible means the following connect immediately:

| Client | How |
|---|---|
| **Cherry Studio** | Provider → OpenAI-compatible → API URL: `http://127.0.0.1:8080/v1` → Key: `anything` → Model: manual add |
| **LibreChat / Open WebUI** | add a custom provider with the endpoint |
| **Python OpenAI SDK** | as shown above |
| **curl / any HTTP tool** | as shown above |

---

## 6.5 Multi-chat & Sessions (concurrent users)

```bash
.\joule.ps1 serve models/Qwen3-30B --port 8080 --max-concurrent 4
```

- **--max-concurrent N**: bounded worker pool — N users chat concurrently
  (the aggregate-throughput path; weight reads amortize per batch).
- **session_id** in the request body: per-user conversation identity →
  `GET /sessions` lists active sessions (id, message count, activity).
- The chat UI generates a per-tab session id automatically (shown in the header).

```json
POST /v1/chat/completions
{ "messages": [...], "session_id": "user-abc" }
```

---

## 6. Cache — instant answers for repeats

- Serve keeps an **answer cache** in the model folder at `answer_cache.json`
- An exact (or normalized) repeat question → **instant answer** (a few ms, no model compute)
- To clear the cache: delete `answer_cache.json` while the server is stopped

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: jouleai` | use the `\joule.ps1` launcher from the repo root (it sets PYTHONPATH) |
| Port in use (`Address already in use`) | change with `--port 8081`, or stop the old server |
| First query very slow | pool warm-up — later queries are faster |
| Not enough RAM | lower `--budget-gb` (a larger budget makes decode faster — tune to your machine) |
| Defender blocks a DLL | add `src/jouleai/native/*.dll` to exclusions (DLLs are dependency-free and normally not blocked) |
| Serving a dense model | today serve supports MoE models only — dense serve wiring is next |

---

## 8. Expected performance (31 GB Ryzen AI laptop, Qwen3-30B Q4)

| Item | Value |
|---|---|
| First token (warm, fused prefill) | ~2 s (Entry 68 batched prefill; cold ~20 s) |
| Decode single-stream (native batch kernel, B=1) | **~7-10 tok/s** (bandwidth floor, Entry 50) |
| Decode aggregate (batch B=8, kernel bench) | **19.6 tok/s** with dummy tokens; **real serve ~3-5 tok/s** (Entry 70) |
| Repeat question (cache hit) | **<20 ms** |
| Disk | 15.4 GB (Q4) |
| RAM | **fixed weights 1.87 GB** (int8); working set 8-13 GB with experts (Entry 51) |
| Quality (MoE) | **budget-invariant** (identical across RAM budgets); Q4 long-gen drifts (Entry 73) |
| Quality (dense sparse, future) | **verified approximate** (verify gate + fallback — NOT lossless) |
| Privacy | everything local — nothing leaves the machine |

> **Honest physics**: single-stream decode is memory-bandwidth-bound — ~7-10
> tok/s is the floor for 30B-A3B Q4 on this laptop (no kernel can change that).
> Higher throughput is **aggregate across concurrent users** — with batch B the
> weights are read once per B tokens, but the measured real serve aggregate is
> ~3-5 tok/s (19.6 @ B=8 is a dummy-token kernel benchmark, Entry 70).

---

## 9. Roadmap (next steps)

- Dense models (Llama/Mistral/Qwen3-8B) serve wiring — dense quantization + verify-gated sparse decode
- Generic converter: any HF model (Kimi/GLM/DeepSeek) convert + serve
- Shape-generic native kernel (dynamic workspace — any model dimensions)
- True disk-backed loading (mmap-only, no pre-touch — RAM ∝ working set)
- int8 VNNI GEMM (QKV + lm_head) + cache-blocked FFN → higher aggregate
- Speculative decoding with a **close-in-size same-family draft** (0.6B was
  tested and rejected — Entry 69 acceptance ~0; a 1.7B/4B draft for the 30B
  target is the next try)
- Ollama-compatible endpoints + MCP
