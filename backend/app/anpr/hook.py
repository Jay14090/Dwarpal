"""Per-camera vehicle branch: plate detect -> OCR -> normalize -> multi-frame vote -> decision.

Realtime cameras run the detector/OCR models; cached cameras replay raw OCR reads recorded by
`make index` (plates.jsonl) through the same normalization and voting, so decisions, registry
matches and events are produced live in both modes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.anpr.normalize import normalize
from app.anpr.ocr import PlateOcr
from app.anpr.plate_detector import PlateDetector
from app.anpr.voting import PlateVote, PlateVoter
from app.core.config import AnprSettings
from app.pipeline.frames import FrameResult, Track


@dataclass(frozen=True)
class RawRead:
    track_id: int
    text: str
    conf: float
    min_char_conf: float
    plate_xyxy: tuple[float, float, float, float]  # frame pixels


@dataclass
class PlateDecision:
    camera_id: str
    track_id: int
    vehicle_label: str
    vote: PlateVote
    ts: float
    frame: int
    thumb_jpeg: bytes | None
    extra: dict[str, Any] = field(default_factory=dict)


def plates_cache_path(cache_dir: Path, camera_id: str) -> Path:
    return cache_dir / camera_id / "plates.jsonl"


class PlateReadCache:
    def __init__(self, path: Path) -> None:
        self.by_frame: dict[int, list[RawRead]] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            self.by_frame.setdefault(d["frame"], []).append(
                RawRead(
                    d["track_id"], d["text"], d["conf"], d["min_char_conf"], tuple(d["plate_xyxy"])
                )  # type: ignore[arg-type]
            )

    def at(self, frame: int) -> list[RawRead]:
        return self.by_frame.get(frame, [])


class PlateReadRecorder:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, frame: int, r: RawRead) -> None:
        self.lines.append(json.dumps({"frame": frame, "track_id": r.track_id, "text": r.text, "conf": round(r.conf, 4),
                                      "min_char_conf": round(r.min_char_conf, 4), "plate_xyxy": [round(v, 1) for v in r.plate_xyxy]}))  # fmt: skip

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + ("\n" if self.lines else ""))


def expand(
    xyxy: tuple[float, float, float, float], shape: tuple[int, ...], frac: float = 0.05
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    H, W = shape[:2]
    return (
        max(0, int(x1 - frac * w)),
        max(0, int(y1 - frac * h)),
        min(W, int(x2 + frac * w)),
        min(H, int(y2 + frac * h)),
    )


class PlateHook:
    def __init__(
        self,
        camera_id: str,
        cfg: AnprSettings,
        on_decision: Callable[[PlateDecision], str | None],
        *,
        detector: PlateDetector | None = None,
        ocr: PlateOcr | None = None,
        cache: PlateReadCache | None = None,
        recorder: Callable[[int, RawRead], None] | None = None,
        grace_frames: int = 45,
    ) -> None:
        if cache is None and (detector is None or ocr is None):
            raise ValueError(f"{camera_id}: PlateHook needs detector + OCR, or a plate read cache")
        self.camera_id = camera_id
        self.cfg = cfg
        self.on_decision = on_decision
        self.detector = detector
        self.ocr = ocr
        self.cache = cache
        self.recorder = recorder
        self.grace_frames = grace_frames
        self.voter = PlateVoter(cfg.min_reads, cfg.min_share, cfg.max_reads)
        self._last_read: dict[int, int] = {}
        self._last_seen: dict[int, tuple[int, str]] = {}  # track -> (frame, label)
        self._best_thumb: dict[int, tuple[float, bytes]] = {}
        self.decisions: dict[int, str] = {}  # track -> plate label shown on the stream

    def _raw_reads(self, result: FrameResult, vehicles: list[Track]) -> list[RawRead]:
        f = result.frame.index
        due = [
            t
            for t in vehicles
            if (last := self._last_read.get(t.track_id)) is None
            or f - last >= self.cfg.sample_every
            or f < last
        ]
        if not due or self.detector is None or self.ocr is None:
            return []
        img = result.frame.image
        crops, meta = [], []
        for t in due:
            self._last_read[t.track_id] = f
            x1, y1, x2, y2 = expand(t.xyxy, img.shape)
            region = img[y1:y2, x1:x2]
            if region.size == 0:
                continue
            boxes = self.detector.detect(region)
            if not boxes:
                continue
            bx1, by1, bx2, by2 = boxes[0].xyxy
            if by2 - by1 < self.cfg.min_plate_height_px:
                continue
            plate = region[int(by1) : int(by2), int(bx1) : int(bx2)]
            if plate.size == 0:
                continue
            crops.append(plate)
            meta.append((t.track_id, (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)))
        return [
            RawRead(tid, r.text, r.conf, r.min_char_conf, box)
            for (tid, box), r in zip(meta, self.ocr.read(crops), strict=True)
        ]

    def _thumb(self, image: np.ndarray, xyxy: tuple[float, float, float, float]) -> bytes | None:
        x1, y1, x2, y2 = expand(xyxy, image.shape, 0.15)
        ok, buf = cv2.imencode(".jpg", image[y1:y2, x1:x2])
        return buf.tobytes() if ok else None

    def _decide(self, track_id: int, label: str, vote: PlateVote, result: FrameResult) -> None:
        thumb = self._best_thumb.pop(track_id, (0.0, None))[1]
        shown = self.on_decision(
            PlateDecision(
                self.camera_id, track_id, label, vote, result.frame.ts, result.frame.index, thumb
            )
        )
        self.decisions[track_id] = shown or vote.plate

    def __call__(self, result: FrameResult) -> None:
        f = result.frame.index
        vehicles = [t for t in result.tracks if t.label in self.cfg.vehicle_labels]
        labels = {t.track_id: t.label for t in vehicles}
        reads = self.cache.at(f) if self.cache is not None else self._raw_reads(result, vehicles)
        for r in reads:
            if self.recorder is not None:
                self.recorder(f, r)
            if r.min_char_conf < self.cfg.min_char_conf:
                continue
            norm = normalize(r.text)
            if norm.valid:
                best = self._best_thumb.get(r.track_id)
                if best is None or r.conf > best[0]:
                    thumb = self._thumb(result.frame.image, r.plate_xyxy)
                    if thumb is not None:
                        self._best_thumb[r.track_id] = (r.conf, thumb)
            vote = self.voter.add(r.track_id, norm, r.conf)
            if vote is not None:
                self._decide(r.track_id, labels.get(r.track_id, "vehicle"), vote, result)
        for t in vehicles:
            self._last_seen[t.track_id] = (f, t.label)
            if t.track_id in self.decisions:
                t.extra["plate"] = self.decisions[t.track_id]
        # tracks gone for longer than the tracker keeps them: settle their vote
        for tid, (last, label) in list(self._last_seen.items()):
            if f - last > self.grace_frames or f < last:
                del self._last_seen[tid]
                self._last_read.pop(tid, None)
                self.decisions.pop(tid, None)
                vote = self.voter.finish(tid)
                if vote is not None:
                    self._decide(tid, label, vote, result)
                self._best_thumb.pop(tid, None)
