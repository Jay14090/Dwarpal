"""Background Postgres writer for the engine (plate reads, events; tracks/embeddings in P6).

Inference threads hand over small callables; one writer thread runs them in their own
transaction, so a slow or unavailable database never stalls video processing.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.core.config import Camera, IndexSettings
from app.datasets.common import Calibration
from app.pipeline.attributes import ColorVote
from app.pipeline.frames import FrameResult
from app.pipeline.height import HeightEstimate
from app.pipeline.reid import crop, crop_quality, l2n
from app.zones import zones_at

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


# --------------------------------------------------------------------------- track indexing (P6)


class ImageEncoder(Protocol):
    def embed_images(self, crops: list[np.ndarray]) -> np.ndarray: ...


@dataclass
class TrackRecord:
    camera_id: str
    track_id: int
    first_ts: float
    last_ts: float
    first_frame: int
    last_frame: int
    global_id: int | None = None
    role: str = "pending"
    crops: list[tuple[float, np.ndarray]] = field(default_factory=list)
    body: list[np.ndarray] = field(default_factory=list)
    zones: set[str] = field(default_factory=set)
    height: HeightEstimate = field(default_factory=HeightEstimate)
    last_crop_frame: int = -(10**9)


@dataclass
class FinishedTrack:
    camera_id: str
    local_track_id: int
    global_id: int | None
    role: str
    start_ts: float
    end_ts: float
    start_frame: int
    end_frame: int
    upper_color: str | None
    lower_color: str | None
    height_cm: float | None
    height_err_cm: float | None
    quality: float
    zones: list[str]
    clip: np.ndarray | None
    body: np.ndarray | None
    thumb_jpeg: bytes | None


class TrackIndexer:
    """FrameHook that turns finished person tracks into searchable records."""

    def __init__(
        self,
        camera: Camera,
        cfg: IndexSettings,
        sink: Callable[[FinishedTrack], None],
        *,
        clip: ImageEncoder | None = None,
        clip_top_k: int = 3,
        calibration: Calibration | None = None,
        grace_frames: int = 45,
    ) -> None:
        self.camera = camera
        self.cfg = cfg
        self.sink = sink
        self.clip = clip
        self.clip_top_k = clip_top_k
        self.calibration = calibration
        self.grace_frames = grace_frames
        self.records: dict[int, TrackRecord] = {}

    def __call__(self, result: FrameResult) -> None:
        f, ts = result.frame.index, result.frame.ts
        img = result.frame.image
        h, w = img.shape[:2]
        persons = [t for t in result.tracks if t.is_person]
        for t in persons:
            rec = self.records.get(t.track_id)
            if rec is not None and (
                f < rec.last_frame or ts - rec.first_ts > self.cfg.max_track_seconds
            ):
                self._finish(rec)  # video looped or a long stay: close this segment
                rec = None
            if rec is None:
                rec = TrackRecord(self.camera.id, t.track_id, ts, ts, f, f)
                self.records[t.track_id] = rec
            rec.last_ts, rec.last_frame = ts, f
            if t.global_id is not None:
                rec.global_id = t.global_id
                rec.role = t.role
            rec.zones.update(zones_at(self.camera, t.xyxy, w, h))
            rec.height.add(self.calibration, t.xyxy)
            body = t.extra.get("body_sample")
            if body is not None and len(rec.body) < 30:
                rec.body.append(body[1])
            if f - rec.last_crop_frame >= self.cfg.crop_every and img.shape[0] > 1:
                rec.last_crop_frame = f
                q = crop_quality(t.xyxy, t.conf, (w, h), [o.xyxy for o in persons if o is not t])
                if q > 0 and (len(rec.crops) < self.cfg.crops_per_track or q > rec.crops[-1][0]):
                    c = crop(img, t.xyxy)
                    if c.size:
                        rec.crops.append((q, c.copy()))
                        rec.crops.sort(key=lambda x: x[0], reverse=True)
                        del rec.crops[self.cfg.crops_per_track :]
        live = {t.track_id for t in persons}
        for tid, rec in list(self.records.items()):
            if tid not in live and (f - rec.last_frame > self.grace_frames or f < rec.last_frame):
                self._finish(rec)

    def flush(self) -> None:
        for rec in list(self.records.values()):
            self._finish(rec)

    def _finish(self, rec: TrackRecord) -> None:
        self.records.pop(rec.track_id, None)
        if rec.last_ts - rec.first_ts < self.cfg.min_track_seconds or not rec.crops:
            return
        colors = ColorVote()
        for q, c in rec.crops:
            colors.add(c, q)
        upper, lower, _, _ = colors.result()
        clip = None
        if self.clip is not None:
            emb = self.clip.embed_images([c for _, c in rec.crops[: self.clip_top_k]])
            clip = l2n(emb.mean(0))
        height, err = rec.height.result()
        best = rec.crops[0][1]
        scale = self.cfg.thumb_height / max(1, best.shape[0])
        thumb = cv2.resize(best, (max(1, int(best.shape[1] * scale)), self.cfg.thumb_height))
        ok, buf = cv2.imencode(".jpg", thumb)
        self.sink(
            FinishedTrack(
                camera_id=rec.camera_id, local_track_id=rec.track_id, global_id=rec.global_id, role=rec.role,
                start_ts=rec.first_ts, end_ts=rec.last_ts, start_frame=rec.first_frame, end_frame=rec.last_frame,
                upper_color=upper, lower_color=lower, height_cm=height, height_err_cm=err,
                quality=round(rec.crops[0][0], 3), zones=sorted(rec.zones), clip=clip,
                body=l2n(np.mean(rec.body, 0)) if rec.body else None, thumb_jpeg=buf.tobytes() if ok else None,
            )
        )  # fmt: skip
