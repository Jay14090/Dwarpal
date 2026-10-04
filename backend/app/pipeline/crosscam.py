"""Per-camera hook that feeds person tracks into the shared GlobalTracker.

For each frame: pick person tracks that are due for an appearance sample (every
`sample_every` frames, quality >= min_quality), get their embeddings (computed by the
encoder in realtime/index mode, or looked up from reid.npz in cached mode), project foot
points to the ground plane when the camera is calibrated, then observe/assign global ids.

The same class records embeddings during `make index`, so cached replay samples exactly the
frames that realtime mode would have sampled.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import ReidSettings
from app.datasets.common import Calibration
from app.pipeline.frames import FrameResult, Track
from app.pipeline.global_tracker import GlobalTracker
from app.pipeline.reid import ReidEncoder, crop, crop_quality

Recorder = Callable[[int, int, float, np.ndarray], None]  # (frame, track_id, quality, emb)


def reid_cache_path(cache_dir: Path, camera_id: str) -> Path:
    return cache_dir / camera_id / "reid.npz"


@dataclass
class ReidCacheWriter:
    rows: list[tuple[int, int, float]] = field(default_factory=list)
    embs: list[np.ndarray] = field(default_factory=list)

    def __call__(self, frame: int, track_id: int, quality: float, emb: np.ndarray) -> None:
        self.rows.append((frame, track_id, quality))
        self.embs.append(emb.astype(np.float16))

    def save(self, path: Path, meta: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        order = sorted(range(len(self.rows)), key=lambda i: self.rows[i][:2])
        rows = np.array([self.rows[i] for i in order], np.float64).reshape(-1, 3)
        dim = self.embs[0].shape[0] if self.embs else 512
        embs = np.stack([self.embs[i] for i in order]) if order else np.zeros((0, dim), np.float16)
        np.savez_compressed(
            path,
            frame=rows[:, 0].astype(np.int32),
            track_id=rows[:, 1].astype(np.int32),
            quality=rows[:, 2].astype(np.float32),
            emb=embs,
            meta=np.array(json.dumps(meta)),
        )


class ReidCache:
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


def foot_point(t: Track) -> tuple[float, float]:
    x1, _, x2, y2 = t.xyxy
    return (x1 + x2) / 2, y2


class CrossCameraHook:
    def __init__(
        self,
        camera_id: str,
        cfg: ReidSettings,
        tracker: GlobalTracker | None,
        *,
        encoder: ReidEncoder | None = None,
        cache: ReidCache | None = None,
        calibration: Calibration | None = None,
        recorder: Recorder | None = None,
        grace_s: float = 2.0,
    ) -> None:
        if encoder is None and cache is None:
            raise ValueError(f"{camera_id}: CrossCameraHook needs an encoder or a reid cache")
        self.camera_id = camera_id
        self.cfg = cfg
        self.tracker = tracker
        self.encoder = encoder
        self.cache = cache
        self.calibration = calibration
        self.recorder = recorder
        self.grace_s = grace_s
        self._last_sample: dict[int, int] = {}

    def due(self, result: FrameResult) -> list[tuple[Track, float]]:
        """Person tracks to sample this frame, with their crop quality."""
        h, w = result.frame.image.shape[:2]
        persons = [t for t in result.tracks if t.is_person]
        out = []
        for t in persons:
            last = self._last_sample.get(t.track_id)
            if (
                last is not None
                and result.frame.index - last < self.cfg.sample_every
                and result.frame.index >= last
            ):
                continue
            others = [o.xyxy for o in persons if o is not t]
            q = crop_quality(t.xyxy, t.conf, (w, h), others)
            if q >= self.cfg.min_quality:
                out.append((t, q))
        return out

    def embeddings(
        self, result: FrameResult, due: list[tuple[Track, float]]
    ) -> dict[int, tuple[float, np.ndarray]]:
        if self.cache is not None:
            return self.cache.at(result.frame.index)
        if not due or self.encoder is None:
            return {}
        embs = self.encoder.embed([crop(result.frame.image, t.xyxy) for t, _ in due])
        return {t.track_id: (q, e) for (t, q), e in zip(due, embs, strict=True)}

    def world_xy(self, t: Track) -> tuple[float, float] | None:
        if self.calibration is None:
            return None
        xy = self.calibration.image_to_ground(np.array([foot_point(t)]))[0]
        return float(xy[0]), float(xy[1])

    def __call__(self, result: FrameResult) -> None:
        due = self.due(result) if self.cache is None else []
        samples = self.embeddings(result, due)
        for tid in samples:
            self._last_sample[tid] = result.frame.index
        if self.recorder is not None:
            for tid, (q, e) in samples.items():
                self.recorder(result.frame.index, tid, q, e)
        if self.tracker is None:
            return
        ts = result.frame.ts
        live = set()
        for t in result.tracks:
            if not t.is_person:
                continue
            live.add(t.track_id)
            q, e = samples.get(t.track_id, (0.0, None))
            xy = self.world_xy(t)
            t.extra["world_xy"] = xy
            t.global_id = self.tracker.observe(self.camera_id, t.track_id, ts, e, q, xy)
        self.tracker.end_stale(self.camera_id, live, ts, self.grace_s)
        for tid in [k for k in self._last_sample if k not in live]:
            if result.frame.index - self._last_sample[tid] > 10 * self.cfg.sample_every:
                del self._last_sample[tid]
