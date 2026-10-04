"""Plate detector interface + implementations.

`oim`  : open-image-models YOLOv9 license-plate detectors (ONNX, pretrained, GitHub release assets)
`yolo` : Ultralytics weights fine-tuned on Indian plates (notebooks/train_plate.ipynb)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.core.config import PlateDetectorSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlateBox:
    xyxy: tuple[float, float, float, float]
    conf: float


class PlateDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[PlateBox]:
        """BGR image -> plate boxes in image pixels, best first."""
        ...


class OimPlateDetector:
    def __init__(self, cfg: PlateDetectorSettings, device: str) -> None:
        from open_image_models import create_detector

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.model = create_detector(cfg.model, conf_thresh=cfg.conf, providers=providers)
        log.info("plate detector %s on %s", cfg.model, device)

    def detect(self, image: np.ndarray) -> list[PlateBox]:
        out = [
            PlateBox(tuple(float(v) for v in r.bounding_box.xyxy), float(r.confidence))  # type: ignore[arg-type]
            for r in self.model.predict(image)
        ]
        return sorted(out, key=lambda b: b.conf, reverse=True)


class YoloPlateDetector:
    def __init__(self, cfg: PlateDetectorSettings, device: str) -> None:
        from app.pipeline.ultra import yolo

        if cfg.weights is None or not cfg.weights.is_file():
            raise FileNotFoundError(
                f"plate detector weights {cfg.weights} not found (train with notebooks/train_plate.ipynb)"
            )
        self.model = yolo(str(cfg.weights))
        self.conf = cfg.conf
        self.device = device

    def detect(self, image: np.ndarray) -> list[PlateBox]:
        r = self.model.predict(image, conf=self.conf, device=self.device, verbose=False)[0]
        boxes = r.boxes.cpu().numpy()
        out = [
            PlateBox(tuple(float(v) for v in b), float(c))
            for b, c in zip(boxes.xyxy, boxes.conf, strict=True)
        ]  # type: ignore[arg-type]
        return sorted(out, key=lambda b: b.conf, reverse=True)


def build_plate_detector(cfg: PlateDetectorSettings, device: str) -> PlateDetector:
    return OimPlateDetector(cfg, device) if cfg.backend == "oim" else YoloPlateDetector(cfg, device)
