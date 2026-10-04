"""Faces: find the face inside a person box and embed it (InsightFace buffalo_l: SCRFD + ArcFace).

Only the head region of person crops that are tall enough is searched, which keeps the cost
proportional to the number of *close* people (webcam) rather than to frame size. At CCTV
distance (datasets) faces are usually too small and body Re-ID carries identity instead.
InsightFace pretrained weights are for non-commercial research use only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.core.config import FaceSettings
from app.pipeline.reid import l2n

log = logging.getLogger(__name__)


@dataclass
class FaceSample:
    emb: np.ndarray  # (512,) L2-normalized
    quality: float  # 0..1
    det_score: float
    box: tuple[float, float, float, float]  # face box in frame pixels


class FaceEncoder(Protocol):
    def faces_for(
        self, image: np.ndarray, person_boxes: list[tuple[float, float, float, float]]
    ) -> list[FaceSample | None]:
        """One entry per person box: the best face inside it, or None."""
        ...


def head_region(
    xyxy: tuple[float, float, float, float], shape: tuple[int, ...], frac: float = 0.45
) -> tuple[int, int, int, int]:
    """Upper part of a person box, padded sideways (heads stick out of tight boxes)."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    H, W = shape[:2]
    return (
        max(0, int(x1 - 0.15 * w)),
        max(0, int(y1 - 0.1 * h)),
        min(W, int(x2 + 0.15 * w)),
        min(H, int(y1 + frac * h)),
    )


def face_quality(
    det_score: float, face_h: float, kps: np.ndarray | None, cfg: FaceSettings
) -> float:
    """det score x size ramp x frontalness (nose between the eyes => looking at the camera)."""
    size = float(
        np.clip((face_h - cfg.min_face_px) / max(1, cfg.good_face_px - cfg.min_face_px), 0, 1)
    )
    frontal = 1.0
    if kps is not None and len(kps) >= 3:
        le, re, nose = kps[0], kps[1], kps[2]
        eye_dx = abs(re[0] - le[0])
        if eye_dx > 1e-3:
            offset = abs(nose[0] - (le[0] + re[0]) / 2) / eye_dx  # 0 frontal, ~0.5 profile
            frontal = float(np.clip(1.0 - offset, 0.2, 1.0))
    return float(det_score * size * frontal)


class InsightFaceEncoder:
    def __init__(self, cfg: FaceSettings, device: str) -> None:
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        cfg.root.mkdir(parents=True, exist_ok=True)
        self.app = FaceAnalysis(
            name=cfg.model,
            root=str(cfg.root),
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.app.prepare(
            ctx_id=0 if device == "cuda" else -1,
            det_thresh=cfg.det_thresh,
            det_size=(cfg.det_size, cfg.det_size),
        )
        self.det = self.app.models["detection"]
        self.rec = self.app.models["recognition"]
        self.cfg = cfg
        log.info("faces: %s on %s", cfg.model, device)

    def faces_for(
        self, image: np.ndarray, person_boxes: list[tuple[float, float, float, float]]
    ) -> list[FaceSample | None]:
        from insightface.app.common import Face

        out: list[FaceSample | None] = []
        for box in person_boxes:
            if box[3] - box[1] < self.cfg.min_person_height_px:
                out.append(None)
                continue
            x1, y1, x2, y2 = head_region(box, image.shape)
            region = image[y1:y2, x1:x2]
            if region.size == 0:
                out.append(None)
                continue
            bboxes, kpss = self.det.detect(
                region, input_size=(self.cfg.det_size, self.cfg.det_size), max_num=0
            )
            best: FaceSample | None = None
            for i in range(len(bboxes)):
                fb = bboxes[i]
                kps = kpss[i] if kpss is not None else None
                fh = float(fb[3] - fb[1])
                if fh < self.cfg.min_face_px or kps is None:  # ArcFace aligns on the 5 landmarks
                    continue
                q = face_quality(float(fb[4]), fh, kps, self.cfg)
                if best is not None and q <= best.quality:
                    continue
                face = Face(bbox=fb[:4], kps=kps, det_score=fb[4])
                emb = l2n(np.asarray(self.rec.get(region, face), np.float32))
                best = FaceSample(
                    emb, q, float(fb[4]), (fb[0] + x1, fb[1] + y1, fb[2] + x1, fb[3] + y1)
                )
            out.append(best)
        return out


def build_face_encoder(cfg: FaceSettings, device: str) -> FaceEncoder | None:
    if not cfg.enabled:
        return None
    return InsightFaceEncoder(cfg, device)
