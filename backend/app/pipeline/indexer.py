"""Background Postgres writer for the engine (plate reads, events; tracks/embeddings in P6).

Inference threads hand over small callables; one writer thread runs them in their own
transaction, so a slow or unavailable database never stalls video processing.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

Job = Callable[[Session], Any]


class DbWriter:
    def __init__(self, engine: Engine, max_queue: int = 1000) -> None:
        self.engine = engine
        self._q: queue.Queue[tuple[Job, Callable[[Any], None] | None]] = queue.Queue(
            maxsize=max_queue
        )
        self._stop = threading.Event()
        self.failures = 0
        self.dropped = 0
        self.done = 0
        self._thread = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._thread.start()

    def submit(self, job: Job, on_done: Callable[[Any], None] | None = None) -> None:
        try:
            self._q.put_nowait((job, on_done))
        except queue.Full:
            self.dropped += 1
            log.error("db writer queue full; dropping a write")

    def _run(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            try:
                job, on_done = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                with Session(self.engine, expire_on_commit=False) as s:
                    result = job(s)
                    s.commit()
                self.done += 1
                if on_done is not None:
                    on_done(result)
            except Exception:
                self.failures += 1
                log.exception("db write failed")

    def flush(self, timeout: float = 5.0) -> None:
        """Wait until queued jobs are written (tests, shutdown)."""
        import time

        deadline = time.monotonic() + timeout
        while not self._q.empty() and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.05)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
