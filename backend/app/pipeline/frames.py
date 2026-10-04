"""Plain data containers passed between pipeline stages (numpy-backed, no torch)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Frame:
    camera_id: str
    index: int  # frame number within the source (file position for file sources)
    ts: float  # wall-clock epoch seconds when the frame was read
    image: np.ndarray  # BGR, HxWx3


@dataclass
class Detections:
    """N detections. `cls` holds our label ids (index into labels), not COCO ids."""

    xyxy: np.ndarray  # (N, 4) float32 pixels
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) int32
    labels: tuple[str, ...]

    @classmethod
    def empty(cls, labels: tuple[str, ...]) -> Detections:
        return cls(
            np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32), labels
        )

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: Any) -> Detections:
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask], self.labels)

    # Ultralytics trackers read these attributes.
    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


@dataclass
class Track:
    track_id: int  # unique within the camera
    xyxy: tuple[float, float, float, float]
    conf: float
    label: str  # "person", "car", ...
    # Filled by later stages (P3+): cross-camera identity, role and attributes.
    global_id: int | None = None
    role: str = "pending"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_person(self) -> bool:
        return self.label == "person"


@dataclass
class FrameResult:
    frame: Frame
    tracks: list[Track]
    infer_ms: float = 0.0  # detection + tracking time (0 when replayed from cache)
