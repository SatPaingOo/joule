"""Session manager — multi-chat control for concurrent users.

Per-session isolated conversation context (the KV cache and masks live in the
engine per session), a bounded worker pool (concurrency limit), and a FIFO
queue so N users can chat concurrently while the engine serializes/batches
decode. This is the control plane for aggregate throughput (the only path to
50-150 tok/s: batching amortizes weight reads across B sequences).
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    id: str
    created: float = field(default_factory=time.time)
    messages: list[dict] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)
    # engine-side state (KV cache etc.) kept on the session by the backend


@dataclass
class ChatJob:
    session_id: str
    messages: list[dict]
    max_tokens: int
    stream: bool
    done: threading.Event
    result: dict = None
    error: str = None
    on_token = None


class SessionManager:
    """Owns sessions and serializes generation through a bounded worker pool."""

    def __init__(self, max_concurrent: int = 4, max_sessions: int = 32):
        self.max_concurrent = max_concurrent
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[ChatJob] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._start_workers()

    # ---------------- sessions ----------------
    def get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            if len(self._sessions) >= self.max_sessions:
                # evict least-recently-active
                oldest = min(self._sessions.values(), key=lambda s: s.last_active)
                del self._sessions[oldest.id]
            s = Session(id=session_id or uuid.uuid4().hex[:12])
            self._sessions[s.id] = s
            return s

    def touch(self, sid: str):
        with self._lock:
            if sid in self._sessions:
                self._sessions[sid].last_active = time.time()

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [{"id": s.id, "messages": len(s.messages),
                     "created": round(s.created, 0),
                     "last_active": round(s.last_active, 0)}
                    for s in self._sessions.values()]

    # ---------------- worker pool ----------------
    def _start_workers(self):
        # ONE scheduler thread: it collects jobs and runs them as ONE batch.
        # (max_concurrent threads would each run their own small batch
        # concurrently — racing the C kernel's shared workspace. Batching
        # needs a single collector; max_concurrent now means the max batch
        # size, which is also the concurrency limit.)
        t = threading.Thread(target=self._worker_loop, daemon=True)
        t.start()
        self._workers.append(t)

    DRAIN_TIMEOUT = 0.003   # per extra slot: tiny window so bursts batch up

    def _worker_loop(self):
        # batch scheduler: collect up to max_concurrent jobs, decode them
        # together (one weight read for B tokens) via _on_batch
        while True:
            first = self._queue.get()
            batch = [first]
            # drain up to max_concurrent-1 more without blocking long
            for _ in range(self.max_concurrent - 1):
                try:
                    batch.append(self._queue.get(timeout=self.DRAIN_TIMEOUT))
                except queue.Empty:
                    break
            try:
                if self._on_batch:
                    self._on_batch(batch)
                else:
                    for job in batch:
                        job.result = self._on_job(job)
            except Exception as e:
                for job in batch:
                    if job.result is None:
                        job.error = str(e)
            finally:
                for job in batch:
                    job.done.set()
                    self._queue.task_done()

    # handler injected by serve (calls the engine)
    _on_job = None
    _on_batch = None  # batch handler: receives list[ChatJob], decodes together

    def submit(self, job: ChatJob):
        self._queue.put(job)
        return job

    def wait(self, job: ChatJob, timeout: float = 600.0):
        job.done.wait(timeout)
        if job.error:
            raise RuntimeError(job.error)
        return job.result
