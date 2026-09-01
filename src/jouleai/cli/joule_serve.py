"""joule serve — OpenAI-compatible local serving with the Joule streaming engine.

    python -m jouleai.cli.joule_serve models/Qwen3-30B-A3B-Instruct-2507 --budget-gb 8

Endpoints:
  POST /v1/chat/completions   OpenAI format (stream=true -> SSE)
  GET  /v1/models             model list
  GET  /status                RAM pool, cache, engine stats

Includes the pi* router lite: persistent exact/normalized answer cache
(cache hit -> serve without touching the model).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from jouleai.storage.q4_store import Q4ExpertPool  # noqa: E402
from jouleai.session.session_manager import SessionManager, ChatJob  # noqa: E402


class JouleServer:
    def __init__(self, model_dir: Path, budget_gb: float, precision: str = "q4",
                 native: bool = False):
        self.model_dir = model_dir
        self.native = native
        self.manifest = {}
        mp = model_dir / "joule_manifest.json"
        if mp.exists():
            self.manifest = json.loads(mp.read_text())
        print("loading engine ...", flush=True)
        if native:
            # native path: C prefill + C decode (int8 VNNI on Zen5), no torch engine
            from jouleai.native.decoder3 import NativeDecoder
            self.nd = NativeDecoder(model_dir, max_tokens=4096)
            try:
                self.nd.build_int8_attn()
            except Exception:
                pass  # falls back to bf16 if int8 not built
            cfg_path = model_dir / "config.json"
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            from jouleai.arch.registry import get_spec
            self.spec = get_spec(cfg)
            self.pool = None
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_dir)
            self.eos = self.tok.eos_token_id
            self.cache_path = model_dir / "answer_cache.json"
            self.cache: dict[str, str] = {}
            if self.cache_path.exists():
                self.cache = json.loads(self.cache_path.read_text())
            self.prefill_cache: dict[int, tuple] = {}
            self.cache_hits = 0
            self.requests = 0
            self.engine = "native"
            import threading as _th
            self.lock = _th.Lock()
            self.served = 0
            self.last_tok_s = 0.0
            return
        from jouleai.engine.generic_streamer import GenericStreamer
        from jouleai.arch.registry import get_spec
        cfg_path = model_dir / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists()             else self.manifest.get("config", {})
        self.spec = get_spec(cfg)
        self.eng = GenericStreamer(model_dir)
        if not self.eng.spec.moe:
            raise SystemExit("serve currently supports MoE models "
                             f"(got arch '{self.spec.model_type}'; dense serve is next)")
        self.precision = precision
        if self.precision == "bf16":
            from jouleai.storage.expert_store import Bf16ExpertPool
            self.pool = Bf16ExpertPool(self.eng.store, self.eng.spec.n_layers,
                                       self.eng.spec.n_experts,
                                       int(budget_gb * 1073741824))
        else:
            self.pool = Q4ExpertPool(model_dir, self.eng.spec.n_layers,
                                     self.eng.spec.n_experts,
                                     int(budget_gb * 1073741824), raw=True)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.eos = self.tok.eos_token_id
        self.cache_path = model_dir / "answer_cache.json"
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())
        self.prefill_cache: dict[int, tuple] = {}  # prompt_hash -> (kv_ready_len, out_ids)
        self.cache_hits = 0
        self.served = 0
        self.last_tok_s = 0.0
        self.lock = threading.Lock()
        self.sessions = SessionManager(max_concurrent=4)

    # ---------------- generation ----------------
    def prompt_ids(self, messages: list[dict]):
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=False)
        return self.tok(text, return_tensors="pt")

    def generate(self, messages: list[dict], max_new: int,
                 on_token=None) -> tuple[str, dict]:
        job = ChatJob(session_id="", messages=messages, max_tokens=max_new,
                      stream=on_token is not None, done=threading.Event())
        job.on_token = on_token
        self.sessions._on_job = self._generate_impl
        self.sessions.submit(job)
        return self.sessions.wait(job)

    def _generate_impl(self, job: ChatJob) -> dict:
        if self.native:
            return self._generate_native(job.messages, job.max_tokens,
                                         getattr(job, "on_token", None))
        return self._generate_direct(job.messages, job.max_tokens,
                                     getattr(job, "on_token", None))

    def _generate_batch_native(self, jobs: list):
        """Native batch decode: B prompts batched-prefilled TOGETHER, then all
        decode together via decode_layers_batch (weights read once per B).
        Each job's answer is streamed (on_token deltas) when requested.

        Batched prefill fixes the first-token latency (Entry 34 finding:
        sequential per-job prefill dominated it); batched decode is the
        aggregate-throughput path (Entry 35-48)."""
        jobs = [j for j in jobs if j is not None]
        if not jobs:
            return
        nd = self.nd
        prompts = [self.prompt_ids(j.messages)["input_ids"][0].tolist()
                   for j in jobs]
        try:
            logits_list = nd.prefill_batch(prompts)
        except Exception as e:
            for j in jobs:
                if j.result is None:
                    j.error = str(e)
            return
        states = []
        for i, (j, lg) in enumerate(zip(jobs, logits_list)):
            first = int(lg.argmax())
            states.append({"tok": first, "out": [first],
                           "p_len": len(prompts[i]), "job": j, "done": False,
                           "prev": ""})
        dummy = self.eos if self.eos is not None else 0
        try:
            while any(not s["done"] for s in states):
                toks = [st["tok"] if not st["done"] else dummy for st in states]
                poss = [st["p_len"] + len(st["out"]) - 1 for st in states]
                nxts = nd.decode_batch_argmax(toks, poss)
                for i, st in enumerate(states):
                    if st["done"]:
                        continue
                    st["out"].append(nxts[i])
                    st["tok"] = nxts[i]
                    if nxts[i] != self.eos and len(st["out"]) - 1 < st["job"].max_tokens:
                        if st["job"].on_token:
                            text = self.tok.decode(st["out"][1:], skip_special_tokens=True)
                            delta = text[len(st["prev"]):]
                            if delta:
                                st["job"].on_token(delta)
                                st["prev"] = text
                    else:
                        st["done"] = True
        except Exception as e:
            for st in states:
                if st["job"].result is None:
                    st["job"].error = str(e)
            return
        for st in states:
            ans = self.tok.decode(st["out"][1:], skip_special_tokens=True)
            st["job"].result = (ans, {"tok_s": None, "io_mb_per_tok": 0.0,
                                      "cache_hit": False})
            st["job"].on_token = None

    def _generate_batch(self, jobs: list):
        """Batch decode: B sequences' tokens decoded together (shared weights)."""
        import torch
        # per-job: build KV cache state (session-scoped KV not yet persisted;
        # this prototype decodes each batch's remaining tokens together)
        # Each job has its own prompt prefill + KV. For the batch we decode
        # step-by-step, one batched forward per step (all B together).
        jobs = [j for j in jobs if j is not None]
        if not jobs:
            return
        # per-job prompt prefill (sequential, once) then shared decode
        states = []
        for job in jobs:
            ids = self.prompt_ids(job.messages)
            cache: dict = {}
            with torch.no_grad():
                logits = self.eng.forward(ids, cache, 0, self.pool)
            p_len = ids["input_ids"].shape[1]
            states.append({"cache": cache, "p_len": p_len,
                           "out": [int(logits[0, -1].argmax())],
                           "job": job, "ids": ids})
        # decode loop: batched forward over all active jobs (per-seq positions)
        active = states
        while active:
            toks = torch.tensor([[st["out"][-1]] for st in active])
            positions = [st["p_len"] + len(st["out"]) - 1 for st in active]
            with torch.no_grad():
                logits = self.eng.forward_batch(toks, [st["cache"] for st in active],
                                                positions, self.pool)
            next_active = []
            for i, st in enumerate(active):
                nxt = int(logits[i].argmax())
                st["out"].append(nxt)
                if nxt != self.eos and len(st["out"]) - 1 < st["job"].max_tokens:
                    next_active.append(st)
                else:
                    ans = self.tok.decode(st["out"][1:], skip_special_tokens=True)
                    st["job"].result = (ans, {"tok_s": None, "io_mb_per_tok": 0.0,
                                              "cache_hit": False})
            active = next_active
        # write results for finished jobs
        for st in states:
            if st["job"].result is None:
                ans = self.tok.decode(st["out"][1:], skip_special_tokens=True)
                st["job"].result = (ans, {"tok_s": None, "io_mb_per_tok": 0.0,
                                          "cache_hit": False})
            st["job"].on_token = None

    def _generate_native(self, messages, max_new, on_token) -> tuple[str, dict]:
        """Native path: C prefill + C decode (int8 VNNI), no torch engine."""
        import torch
        key = json.dumps(messages, ensure_ascii=False).strip().lower()
        with self.lock:
            if key in self.cache:
                self.cache_hits += 1
                if on_token:
                    on_token(self.cache[key])
                return self.cache[key], {"cache_hit": True, "tok_s": None,
                                         "io_mb_per_tok": 0.0}
        ids = self.prompt_ids(messages)
        toks = ids["input_ids"][0].tolist()
        t0 = time.perf_counter()
        nd = self.nd
        nd.reset()
        logits = nd.prefill(toks)              # C prefill (lm_head at last token only)
        out = [int(logits.argmax())]
        for i in range(max_new - 1):
            if out[-1] == self.eos:
                break
            out.append(nd.decode_token(out[-1]))
            if on_token:
                text = self.tok.decode(out[1:], skip_special_tokens=True)
                delta = text  # native: emit cumulative (streaming deltas omitted for simplicity)
        wall = time.perf_counter() - t0
        n = max(len(out) - 1, 1)
        answer = self.tok.decode(out[1:], skip_special_tokens=True)
        stats = {"cache_hit": False, "tok_s": round(n / wall, 2), "io_mb_per_tok": 0.0}
        with self.lock:
            self.last_tok_s = stats["tok_s"]
            self.served += 1
            self.cache[key] = answer
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False))
        # release-after-use: drop expert pages beyond the hot set (RAM ∝ working set)
        try:
            self.nd.release(keep_mb=512)
        except Exception:
            pass
        return answer, stats

    def _generate_direct(self, messages, max_new, on_token) -> tuple[str, dict]:
        key = json.dumps(messages, ensure_ascii=False).strip().lower()
        with self.lock:
            if key in self.cache:
                self.cache_hits += 1
                if on_token:
                    on_token(self.cache[key])
                return self.cache[key], {"cache_hit": True, "tok_s": None,
                                         "io_mb_per_tok": 0.0}
        ids = self.prompt_ids(messages)
        cache: dict = {}
        t0 = time.perf_counter()
        logits = self.eng.forward(ids, cache, 0, self.pool)
        p_len = ids["input_ids"].shape[1]
        out = [int(logits[0, -1].argmax())]
        io0 = self.pool.stats.io_bytes
        text_parts: list[str] = []
        prev_text = ""
        for i in range(max_new - 1):
            if out[-1] == self.eos:
                break
            step = torch.tensor([[out[-1]]])
            logits = self.eng.forward(step, cache, p_len + i, self.pool)
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            if on_token:
                text = self.tok.decode(out[1:], skip_special_tokens=True)
                delta = text[len(prev_text):]
                if delta:
                    on_token(delta)
                    prev_text = text
        wall = time.perf_counter() - t0
        n = max(len(out) - 1, 1)
        answer = self.tok.decode(out[1:], skip_special_tokens=True)
        stats = {"cache_hit": False,
                 "tok_s": round(n / wall, 2),
                 "io_mb_per_tok": round((self.pool.stats.io_bytes - io0) / n / 1048576, 1)}
        with self.lock:
            self.last_tok_s = stats["tok_s"]
            self.served += 1
            self.cache[key] = answer
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False))
        return answer, stats

    # ---------------- status ----------------
    def status(self) -> dict:
        import psutil

        return {
            "model": self.model_dir.name,
            "arch": (self.spec.model_type if getattr(self, "spec", None) else "native"),
            "ram_budget_gb": self.manifest.get("budget_gb"),
            "threads": getattr(self, "threads", None),
            "precision": getattr(self, "precision", None),
            "governor": getattr(self, "governor", None),
            "pool_resident_gb": (round(self.pool.resident_bytes() / 1073741824, 2)
                                 if self.pool is not None else 0.0),
            "process_rss_gb": round(psutil.Process().memory_info().rss / 1073741824, 2),
            "cache_entries": len(self.cache),
            "cache_hits": self.cache_hits,
            "requests_served": self.served,
            "last_tok_s": self.last_tok_s,
        }


def _test_page() -> bytes:
    tp = Path(__file__).resolve().parents[3] / "web" / "chat.html"
    html = tp.read_text(encoding="utf-8") if tp.exists() else "<h1>missing web/chat.html</h1>"
    # populate the model dropdown from the models/ directory (any model path)
    models_dir = Path(__file__).resolve().parents[3] / "models"
    opts = []
    if models_dir.exists():
        for d in sorted(p.name for p in models_dir.iterdir()
                        if p.is_dir() and (p / "config.json").exists()):
            opts.append(f'<option value="{d}">{d}</option>')
    if not opts:
        opts = ['<option value="Qwen3-30B-A3B-Instruct-2507">Qwen3-30B-A3B-Instruct-2507</option>']
    html = html.replace("__MODELS__", "\n".join(opts))
    html = html.replace("__MODEL__", "Qwen3-30B-A3B-Instruct-2507")
    return html.encode()


def make_handler(server: JouleServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            print(f"[req] {a[0]}", flush=True)

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "3600")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/v1/models":
                self._json({"object": "list", "data": [
                    {"id": m.name, "object": "model", "owned_by": "joule"}
                    for m in sorted((Path(__file__).resolve().parents[3] / "models").iterdir())
                    if m.is_dir() and (m / "config.json").exists()]})
            elif path.startswith("/v1/model/"):
                # switch the native decoder to another model path
                name = path[len("/v1/model/"):]
                model_path = Path(__file__).resolve().parents[3] / "models" / name
                if not (model_path / "config.json").exists():
                    self._json({"error": f"model not found: {name}"}, 404)
                    return
                try:
                    from jouleai.native.decoder3 import NativeDecoder
                    nd = NativeDecoder(model_path, max_tokens=4096)
                    try:
                        nd.build_int8_attn()
                    except Exception:
                        pass
                    server.nd = nd
                    server.model_dir = model_path
                    server.spec = None
                    try:
                        import json as _j
                        server.tok = server.tok  # tokenizer per model — reload
                        from transformers import AutoTokenizer
                        server.tok = AutoTokenizer.from_pretrained(model_path)
                        server.eos = server.tok.eos_token_id
                    except Exception:
                        pass
                    self._json({"ok": True, "model": name})
                except Exception as e:
                    self._json({"error": str(e)}, 500)
            elif path == "/status":
                self._json(server.status())
            elif path == "/v1/control":
                # the single control-panel view (device + model + plan + health)
                try:
                    self._json(server.control.status())
                except AttributeError:
                    self._json({"error": "control center not attached"})
            elif path == "/sessions":
                self._json({"sessions": server.sessions.snapshot()})
            elif path in ("/", "/test", "/chat"):
                body = _test_page()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path.split("?")[0] != "/v1/chat/completions":
                self._json({"error": "not found"}, 404)
                return
            try:
                req = json.loads(self.rfile.read(int(
                    self.headers.get("Content-Length", 0))))
            except Exception as e:
                self._json({"error": str(e)}, 400)
                return
            messages = req.get("messages", [])
            max_new = int(req.get("max_tokens", 256))
            stream = bool(req.get("stream", False))
            sid = req.get("session_id")
            if sid:
                server.sessions.get_or_create(sid)
                server.sessions.touch(sid)
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            model_id = server.model_dir.name

            if not stream:
                answer, stats = server.generate(messages, max_new)
                self._json({
                    "id": cid, "object": "chat.completion", "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": answer}}],
                    "joule": stats,
                })
                return

            # SSE streaming
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self._cors()
            self.end_headers()

            def emit(delta: str):
                chunk = {"id": cid, "object": "chat.completion.chunk",
                         "created": created, "model": model_id,
                         "choices": [{"index": 0, "delta": {"content": delta},
                                      "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            def on_token(delta: str):
                try:
                    emit(delta)
                except Exception:
                    pass

            answer, stats = server.generate(messages, max_new, on_token=on_token)
            if stats.get("cache_hit"):
                try:
                    emit(answer)
                except Exception:
                    pass
            try:
                end = {"id": cid, "object": "chat.completion.chunk",
                       "created": created, "model": model_id,
                       "choices": [{"index": 0, "delta": {},
                                    "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(end)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8078)
    ap.add_argument("--budget-gb", type=float, default=8.0)
    ap.add_argument("--threads", type=int, default=8,
                    help="expert kernel worker threads (match CPU cores)")
    ap.add_argument("--auto-budget", action="store_true",
                    help="cap RAM budget at 40%% of currently available RAM")
    ap.add_argument("--precision", choices=["q4", "bf16"], default="q4",
                    help="expert store tier: q4 (small/fast-io) or bf16 (full precision)")
    ap.add_argument("--backend", choices=["auto", "native", "pool"], default="auto",
                    help="engine backend: native (fused C kernel) or pool (LRU Q4)")
    ap.add_argument("--profile", choices=["battery", "balanced", "performance"],
                    default="balanced", help="one-shot resource preset")
    ap.add_argument("--max-concurrent", type=int, default=4,
                    help="concurrent chat workers (batching path)")
    args = ap.parse_args()

    # ---- control plane: one place decides how this model runs on this device ----
    import json as _json
    from jouleai.control import plan_for, ModelInfo
    _cfg_path = Path(args.model) / "config.json"
    _c = _json.loads(_cfg_path.read_text()) if _cfg_path.exists() else {}
    _mi = ModelInfo.from_config(_c)
    _overrides = {
        "budget_gb": args.budget_gb if not args.auto_budget else None,
        "threads": args.threads if args.threads != 8 else None,
        "precision": args.precision if args.precision != "q4" else None,
        "backend": args.backend if args.backend != "auto" else None,
        "max_concurrent": args.max_concurrent,
    }
    dev, plan = plan_for(_mi, _overrides)
    budget = plan.budget_gb
    print(f"[control] {dev.os} {dev.total_ram_gb:.0f}GB RAM ({dev.free_ram_gb:.0f}GB free) | "
          f"{dev.cores}c/{dev.logical_cores}t | BW~{dev.mem_bw_gb_s:.0f}GB/s | "
          f"GPU={dev.gpu or 'none'} | NPU={'yes' if dev.npu else 'no'} | tier={dev.tier}")
    print(f"[control] plan: backend={plan.backend} precision={plan.precision} "
          f"budget={budget}GB threads={plan.threads} batch={plan.batch} "
          f"spec={'on' if plan.spec else 'off'}")
    for r in plan.rationale:
        print(f"[control]   - {r}")
    # ---- KV RAM guard: per-session KV is L*maxT*n_kv*hd*2*4 bytes; clamp the
    # batch/concurrency so KV doesn't blow the machine (batch decodes B
    # sessions' KV simultaneously) ----
    try:
        _maxT = 4096  # NativeDecoder default
        _hd = _c.get("head_dim") or (_c.get("hidden_size", 0)
                                     // max(_c.get("num_attention_heads", 1), 1))
        _kv_gb = (_c.get("num_hidden_layers", 0) * _maxT
                  * _c.get("num_key_value_heads", 1) * _hd * 2 * 4) / 1e9
        _max_b = max(1, int(dev.free_ram_gb * 0.30 / max(_kv_gb, 0.1)))
        if plan.max_concurrent > _max_b:
            print(f"[control]   - KV RAM guard: {plan.max_concurrent} -> {_max_b} "
                  f"concurrent (per-session KV {_kv_gb:.2f}GB)")
            plan.max_concurrent = _max_b
    except Exception:
        pass

    native = (args.backend == "native")
    server = JouleServer(Path(args.model), budget, precision=plan.precision,
                         native=native)
    # attach the control center (device + model + plan + live health)
    from jouleai.control import ControlCenter
    server.control = ControlCenter(_mi, _overrides)
    server.sessions = SessionManager(max_concurrent=plan.max_concurrent)
    server.sessions._on_job = server._generate_impl
    server.sessions._on_batch = (server._generate_batch_native if native
                                 else server._generate_batch)
    server.max_concurrent = plan.max_concurrent
    if not native:
        torch.set_num_threads(max(2, plan.threads // 2))  # leave cores for experts
        from concurrent.futures import ThreadPoolExecutor
        server.eng._executor = ThreadPoolExecutor(max_workers=plan.threads)
    server.threads = plan.threads
    server.governor = plan.status()
    st = server.status()
    print(f"engine ready | pool {st['pool_resident_gb']} GB | "
          f"cache {st['cache_entries']} entries | threads {plan.threads}")
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(server))
    print(f"joule serve listening on http://127.0.0.1:{args.port}")
    print("  POST /v1/chat/completions  (OpenAI-compatible, stream supported)")
    print("  GET  /v1/models | /status")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
