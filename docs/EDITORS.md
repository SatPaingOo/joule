# Joule — Editor / IDE Integration Guide (VS Code, Cursor, ...)
> **Context**: editor/dev setup for the Joule repo. See [docs/USAGE.md](USAGE.md)
> (run/API), [docs/STANDARDS.md](STANDARDS.md) (code standards), [docs/JOULE_PAPER.md](JOULE_PAPER.md)
> (the retrospective).

---

> Joule serve exposes an **OpenAI-compatible API** (`http://127.0.0.1:8080/v1`),
> so any tool that speaks OpenAI connects to it.
> Start command:
> `PYTHONPATH=src python -m jouleai.cli.joule_serve models/<model> --port 8080 --budget-gb 8`

---

## 0. Honest status (today)

| Feature | Status |
|---|---|
| **Chat** (Q&A, code explanation, writing) | ✅ works |
| **Streaming** | ✅ works |
| **Tool calling / Function calling** (agent modes) | ❌ not yet — roadmap |
| **FIM autocomplete** (Tab completion) | ❌ not yet — roadmap |
| **Speed** | 30B (native kernel, B=1): **~7-10 tok/s** (bandwidth floor) · batch aggregate: ~3-5 tok/s real serve (kernel bench 19.6 @ B=8, dummy tokens) |

So: **chat is usable today**; agent/autocomplete modes need tool-calling and
FIM endpoints first.

---

## 1. VS Code — Continue.dev (recommended, easiest)

1. Install the "Continue" extension (Continue.dev)
2. Add to `~/.continue/config.yaml`:

```yaml
models:
  - name: Joule-30B
    provider: openai
    apiBase: http://127.0.0.1:8080/v1
    apiKey: joule-local
    model: Qwen3-30B-A3B-Instruct-2507
    roles: [chat, edit]
```

3. In the Continue sidebar, select "Joule-30B" and chat immediately.

## 2. VS Code — Cline / Roo Code (agent style)

1. Install "Cline" (or "Roo Code")
2. Settings → API Provider: **OpenAI Compatible**
   - Base URL: `http://127.0.0.1:8080/v1`
   - API Key: `joule-local`
   - Model ID: `Qwen3-30B-A3B-Instruct-2507`
3. ⚠️ **Note**: Cline/Roo agent mode is based on tool calls — Joule does not
   have tool calling yet, so use **chat/plan modes only**. (Tool calling is on
   the roadmap.)

## 3. VS Code — Twinny (local autocomplete + chat)

1. Install "Twinny"
2. Provider: OpenAI-compatible → `http://127.0.0.1:8080` → select the model id.
3. Autocomplete (FIM) is not supported by Joule yet — use the chat feature.

## 4. Cursor

1. Settings → Models → **OpenAI API Key**: enter any key.
2. Enable **Override OpenAI Base URL** → `http://127.0.0.1:8080/v1`
3. Add a custom model name: `Qwen3-30B-A3B-Instruct-2507` (folder name).
4. In Chat (Cmd/Ctrl+L) select this model.
   ⚠️ Cursor's Tab autocomplete and some Composer/agent features only work
   with their own models (not overridden).

## 5. JetBrains IDEs (PyCharm/IntelliJ)

- Continue.dev's JetBrains plugin (as above), or
- "CodeGPT" plugin → Custom OpenAI-compatible provider → base URL.

## 6. Zed editor

In `settings.json`:

```json
{
  "language_models": {
    "openai": {
      "api_url": "http://127.0.0.1:8080/v1",
      "available_models": [
        { "name": "Qwen3-30B-A3B-Instruct-2507", "max_tokens": 4096 }
      ]
    }
  }
}
```

---

## 7. Model selection recommendations

| Purpose | Model | Why |
|---|---|---|
| Best chat today | `Qwen3-30B-A3B-Instruct-2507` | largest, best quality (budget-invariant MoE) |
| Faster responses | `OLMoE-1B-7B-0824-Instruct` | Q4 3.2 GB, faster decode |
| Coding specialist | `Qwen2.5-Coder-7B` (already downloaded) | best for coding once dense serve is wired |

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| Tool doesn't see the model list | check the Base URL includes `/v1` (`http://127.0.0.1:8080/v1`) |
| Connection refused | check `joule serve` is running and the port (`curl http://127.0.0.1:8080/status`) |
| Cursor won't accept it | the "Override OpenAI Base URL" toggle must be enabled |
| Answers take a while | 30B decode is ~7-10 tok/s (bandwidth floor) — try OLMoE or a smaller model |
