"""Per-camera cache of tracked detections for `run_mode: cached` (written by `make index`).

File: data/cache/<camera>/tracks.npz with parallel arrays, sorted by frame:
    frame (int32), track_id (int32), xyxy (float32 Nx4), conf (float32), label (int8 -> labels)
plus `labels` and `meta` (JSON: source file, frames, fps, detector, created).
Later phases add per-track files next to it (embeddings, attributes).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.pipeline.frames import Track


def cache_path(cache_dir: Path, camera_id: str) -> Path:
    return cache_dir / camera_id / "tracks.npz"


@dataclass
class TrackCacheWriter:
    labels: tuple[str, ...]
    _rows: list[tuple[int, int, float, float, float, float, float, int]] = field(
        default_factory=list
    )

    def add(self, frame: int, tracks: list[Track]) -> None:
        for t in tracks:
            self._rows.append((frame, t.track_id, *t.xyxy, t.conf, self.labels.index(t.label)))

    def save(self, path: Path, meta: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(self._rows, key=lambda r: (r[0], r[1]))
        arr = np.array(rows, dtype=np.float64).reshape(-1, 8)
        np.savez_compressed(
            path,
            frame=arr[:, 0].astype(np.int32),
            track_id=arr[:, 1].astype(np.int32),
            xyxy=arr[:, 2:6].astype(np.float32),
            conf=arr[:, 6].astype(np.float32),
            label=arr[:, 7].astype(np.int8),
            labels=np.array(self.labels),
            meta=np.array(json.dumps(meta)),
        )


class TrackCache:
    """Read-only lookup: tracks for a frame index in O(log n)."""

    def __init__(self, path: Path) -> None:
        with np.load(path) as z:
            self.frame = z["frame"]
            self.track_id = z["track_id"]
            self.xyxy = z["xyxy"]
            self.conf = z["conf"]
            self.label = z["label"]
            self.labels = tuple(str(s) for s in z["labels"])
            self.meta: dict[str, Any] = json.loads(str(z["meta"]))
        self._starts = np.searchsorted(self.frame, np.arange(self.num_frames + 1))

    @property
    def num_frames(self) -> int:
        return int(self.meta.get("num_frames") or (self.frame.max() + 1 if len(self.frame) else 0))

    def tracks_at(self, frame: int) -> list[Track]:
        if frame < 0 or frame >= self.num_frames:
            return []
        lo, hi = self._starts[frame], self._starts[frame + 1]
        return [
            Track(int(self.track_id[i]), tuple(float(v) for v in self.xyxy[i]),  # type: ignore[arg-type]
                  float(self.conf[i]), self.labels[int(self.label[i])])
            for i in range(lo, hi)
        ]  # fmt: skip
