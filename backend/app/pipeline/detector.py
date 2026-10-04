"""Object detector interface and the Ultralytics YOLO implementation."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from app.core.config import DetectorSettings
from app.pipeline.frames import Detections

log = logging.getLogger(__name__)


class Detector(Protocol):
    labels: tuple[str, ...]

    def detect(self, images: list[np.ndarray]) -> list[Detections]: ...


class YoloDetector:
    """Ultralytics YOLO restricted to our classes (person + vehicle types)."""

    def __init__(self, cfg: DetectorSettings, device: str, half: bool = True) -> None:
        from app.pipeline.ultra import yolo  # heavy import, keep it lazy

        self.cfg = cfg
        self.device = device
        self.half = half and device == "cuda"
        self.labels = tuple(cfg.classes)
        self._coco_ids = list(cfg.classes.values())
        self._coco_to_ours = {coco: i for i, coco in enumerate(self._coco_ids)}
        if not cfg.weights.is_file():
            log.info(
                "weights %s not found locally; Ultralytics will download them", cfg.weights.name
            )
            cfg.weights.parent.mkdir(parents=True, exist_ok=True)
        self.model = yolo(str(cfg.weights if cfg.weights.is_file() else cfg.weights.name))
        log.info("detector %s on %s (fp16=%s)", cfg.weights.name, device, self.half)

    def detect(self, images: list[np.ndarray]) -> list[Detections]:
        if not images:
            return []
        results = self.model.predict(
            images,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            classes=self._coco_ids,
            device=self.device,
            quantize=16 if self.half else 32,
            verbose=False,
        )
        out = []
        for r in results:
            boxes = r.boxes.cpu().numpy()
            cls = np.array([self._coco_to_ours[int(c)] for c in boxes.cls], dtype=np.int32)
            out.append(
                Detections(
                    boxes.xyxy.astype(np.float32), boxes.conf.astype(np.float32), cls, self.labels
                )
            )
        return out
