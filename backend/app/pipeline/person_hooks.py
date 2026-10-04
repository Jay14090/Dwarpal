"""Per-camera hooks for faces, identity roles and webcam enrollment capture.

Hook order on a worker: CrossCameraHook (global ids + body samples in track.extra) ->
FaceHook (face samples in track.extra) -> IdentityHook (roles). Samples travel between hooks in
`track.extra["body_sample"]` / `track.extra["face_sample"]` as (quality, embedding).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import FaceSettings
from app.pipeline.face import FaceEncoder
from app.pipeline.frames import FrameResult, Track
from app.pipeline.identity import IdentityEngine
from app.pipeline.reid import ReidEncoder, crop, crop_quality

Recorder = Callable[[int, int, float, np.ndarray], None]


def face_cache_path(cache_dir: Path, camera_id: str) -> Path:
    return cache_dir / camera_id / "faces.npz"


class SampleCache:
    """(frame -> {track_id: (quality, emb)}) loaded from an npz written by SampleCacheWriter."""

    def __init__(self, path: Path) -> None:
        with np.load(path) as z:
            frame, tid, q, emb = z["frame"], z["track_id"], z["quality"], z["emb"]
            self.meta = json.loads(str(z["meta"]))
        self._by_frame: dict[int, dict[int, tuple[float, np.ndarray]]] = {}
        for i in range(len(frame)):
            self._by_frame.setdefault(int(frame[i]), {})[int(tid[i])] = (
                float(q[i]),
                emb[i].astype(np.float32),
            )

    def at(self, frame: int) -> dict[int, tuple[float, np.ndarray]]:
        return self._by_frame.get(frame, {})


class FaceHook:
    def __init__(
        self,
        camera_id: str,
        cfg: FaceSettings,
        *,
        encoder: FaceEncoder | None = None,
        cache: SampleCache | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        if encoder is None and cache is None:
            raise ValueError(f"{camera_id}: FaceHook needs an encoder or a face cache")
        self.camera_id = camera_id
        self.cfg = cfg
        self.encoder = encoder
        self.cache = cache
        self.recorder = recorder
        self._last: dict[int, int] = {}

    def _due(self, result: FrameResult) -> list[Track]:
        f = result.frame.index
        out = []
        for t in result.tracks:
            if not t.is_person or t.xyxy[3] - t.xyxy[1] < self.cfg.min_person_height_px:
                continue
            last = self._last.get(t.track_id)
            if last is None or f - last >= self.cfg.sample_every or f < last:
                out.append(t)
        return out

    def __call__(self, result: FrameResult) -> None:
        if self.cache is not None:
            samples = self.cache.at(result.frame.index)
        else:
            due = self._due(result)
            samples = {}
            if due and self.encoder is not None:
                for t, fs in zip(
                    due,
                    self.encoder.faces_for(result.frame.image, [t.xyxy for t in due]),
                    strict=True,
                ):
                    self._last[t.track_id] = result.frame.index
                    if fs is not None:
                        samples[t.track_id] = (fs.quality, fs.emb)
        for t in result.tracks:
            if t.track_id in samples and t.is_person:
                t.extra["face_sample"] = samples[t.track_id]
        if self.recorder is not None:
            for tid, (q, e) in samples.items():
                self.recorder(result.frame.index, tid, q, e)


class IdentityHook:
    """Feeds samples of tracks with a global id to the shared IdentityEngine; sets track.role."""

    def __init__(self, identity: IdentityEngine) -> None:
        self.identity = identity

    def __call__(self, result: FrameResult) -> None:
        ts = result.frame.ts
        for t in result.tracks:
            if not t.is_person:
                continue
            if t.global_id is None:
                t.role = "pending"
                continue
            for kind, key in (("face", "face_sample"), ("body", "body_sample")):
                sample = t.extra.get(key)
                if sample is not None:
                    self.identity.observe(t.global_id, kind, sample[1], sample[0], ts)  # type: ignore[arg-type]
            t.role = self.identity.role(t.global_id)


# --------------------------------------------------------------------------- enrollment capture


@dataclass
class Shot:
    quality: float
    emb: np.ndarray
    ts: float


@dataclass
class EnrollmentCollector:
    """Collects the best face/body shots of the most prominent person for `seconds`.

    Attached to a camera worker on request; computes embeddings itself at <= `rate_hz` so the
    capture doesn't depend on the regular sampling schedule.
    """

    seconds: float
    shots: int
    face_encoder: FaceEncoder | None
    body_encoder: ReidEncoder | None
    on_done: Callable[[dict[str, Any]], None]
    rate_hz: float = 5.0
    min_gap_s: float = 0.3
    started: float = field(default_factory=time.monotonic)
    faces: list[Shot] = field(default_factory=list)
    bodies: list[Shot] = field(default_factory=list)
    thumb: tuple[float, bytes] | None = None
    frames_seen: int = 0
    _last: float = 0.0
    _done: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _keep(shots: list[Shot], new: Shot, n: int, min_gap: float) -> None:
        clash = [s for s in shots if abs(s.ts - new.ts) < min_gap]
        if clash:
            worst = min(clash, key=lambda s: s.quality)
            if worst.quality >= new.quality:
                return
            shots.remove(worst)
        shots.append(new)
        shots.sort(key=lambda s: s.quality, reverse=True)
        del shots[n:]

    def __call__(self, result: FrameResult) -> None:
        with self._lock:
            if self._done:
                return
            now = time.monotonic()
            if now - self.started > self.seconds:
                self._finish()
                return
            if now - self._last < 1.0 / self.rate_hz:
                return
            self._last = now
            persons = [t for t in result.tracks if t.is_person]
            if not persons:
                return
            self.frames_seen += 1
            t = max(persons, key=lambda p: (p.xyxy[2] - p.xyxy[0]) * (p.xyxy[3] - p.xyxy[1]))
            img = result.frame.image
            h, w = img.shape[:2]
            if self.face_encoder is not None:
                fs = self.face_encoder.faces_for(img, [t.xyxy])[0]
                if fs is not None:
                    self._keep(
                        self.faces, Shot(fs.quality, fs.emb, now), self.shots, self.min_gap_s
                    )
                    if self.thumb is None or fs.quality > self.thumb[0]:
                        x1, y1, x2, y2 = (int(v) for v in fs.box)
                        pad = int(0.3 * (y2 - y1))
                        face_img = img[max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad]
                        ok, buf = cv2.imencode(".jpg", face_img)
                        if ok:
                            self.thumb = (fs.quality, buf.tobytes())
            if self.body_encoder is not None:
                q = crop_quality(t.xyxy, t.conf, (w, h))
                emb = self.body_encoder.embed([crop(img, t.xyxy)])[0]
                self._keep(self.bodies, Shot(max(q, 1e-3), emb, now), self.shots, self.min_gap_s)

    def expire(self) -> None:
        """Called periodically by the engine so a capture ends even if no frames arrive."""
        with self._lock:
            if not self._done and time.monotonic() - self.started > self.seconds:
                self._finish()

    def _finish(self) -> None:
        self._done = True
        self.on_done(
            {
                "face": [s.emb.tolist() for s in self.faces],
                "face_quality": [round(s.quality, 3) for s in self.faces],
                "body": [s.emb.tolist() for s in self.bodies],
                "body_quality": [round(s.quality, 3) for s in self.bodies],
                "frames_with_person": self.frames_seen,
                "thumb_jpeg": self.thumb[1] if self.thumb else None,
            }
        )

    @property
    def done(self) -> bool:
        return self._done
