"""Per-camera multi-object tracker (ByteTrack from Ultralytics) behind a small interface.

People and vehicles get separate ByteTrack instances so a person is never associated with a
car box; both draw ids from one per-camera counter so ids stay unique within the camera.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Protocol

import numpy as np

from app.core.config import TrackerSettings
from app.pipeline.frames import Detections, Track


class Tracker(Protocol):
    def update(self, dets: Detections, image: np.ndarray | None = None) -> list[Track]: ...


class ByteTracker:
    def __init__(self, cfg: TrackerSettings, min_box_height: float = 0.0) -> None:
        from app.pipeline.ultra import disable_telemetry

        disable_telemetry()
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = SimpleNamespace(**cfg.model_dump(exclude={"type"}))
        self._groups = {"person": BYTETracker(args), "vehicle": BYTETracker(args)}
        self.min_box_height = min_box_height
        self._next_id = itertools.count(1)
        # (group, ultralytics id) -> our id. Ultralytics ids come from a process-global counter.
        self._ids: dict[tuple[str, int], int] = {}

    def update(self, dets: Detections, image: np.ndarray | None = None) -> list[Track]:
        if len(dets):
            h = dets.xyxy[:, 3] - dets.xyxy[:, 1]
            dets = dets[h >= self.min_box_height]
        person_idx = dets.labels.index("person") if "person" in dets.labels else -1
        is_person = dets.cls == person_idx
        tracks: list[Track] = []
        for group, mask in (("person", is_person), ("vehicle", ~is_person)):
            sub = dets[mask]
            rows = self._groups[group].update(sub, image)
            for x1, y1, x2, y2, tid, score, cls_, _idx in rows:
                key = (group, int(tid))
                if key not in self._ids:
                    self._ids[key] = next(self._next_id)
                label = dets.labels[int(cls_)]
                tracks.append(Track(self._ids[key], (x1, y1, x2, y2), float(score), label))
        return tracks
