"""Plate OCR interface + fast-plate-ocr implementation (CCT models, ONNX).

The pretrained global models do not list India among their training regions; Indian plates use
the same Latin alphanumerics and fit the 10 output slots, and notebooks/train_plate.ipynb
fine-tunes on Indian plates when labeled data is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from app.core.config import PlateOcrSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrRead:
    text: str
    conf: float  # mean per-character confidence over the decoded characters
    min_char_conf: float


class PlateOcr(Protocol):
    def read(self, plates: list[np.ndarray]) -> list[OcrRead]:
        """BGR plate crops -> raw text reads."""
        ...


class FastPlateOcr:
    def __init__(self, cfg: PlateOcrSettings, device: str) -> None:
        from fast_plate_ocr import LicensePlateRecognizer

        kwargs: dict = {"device": "cuda" if device == "cuda" else "cpu"}
        if cfg.onnx_path is not None:
            kwargs.update(
                onnx_model_path=str(cfg.onnx_path), plate_config_path=str(cfg.config_path)
            )
        else:
            kwargs["hub_ocr_model"] = cfg.model
        self.model = LicensePlateRecognizer(**kwargs)
        self.rgb = self.model.config.image_color_mode == "rgb"
        log.info(
            "plate OCR %s (%s input)", cfg.onnx_path or cfg.model, "rgb" if self.rgb else "gray"
        )

    def read(self, plates: list[np.ndarray]) -> list[OcrRead]:
        if not plates:
            return []
        imgs = [
            cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            if self.rgb
            else cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)[:, :, None]
            for p in plates
        ]
        out = []
        for pred in self.model.run(imgs, return_confidence=True):
            n = len(pred.plate)
            probs = (
                np.asarray(pred.char_probs, float)[:n]
                if pred.char_probs is not None and n
                else np.zeros(0)
            )
            out.append(
                OcrRead(
                    pred.plate, float(probs.mean()) if n else 0.0, float(probs.min()) if n else 0.0
                )
            )
        return out


def build_plate_ocr(cfg: PlateOcrSettings, device: str) -> PlateOcr:
    return FastPlateOcr(cfg, device)
