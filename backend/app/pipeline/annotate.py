"""Draw tracks on frames: boxes colored by role (resident/staff/unknown/pending) or vehicle."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.config import AnnotateSettings
from app.pipeline.frames import Track


class Annotator:
    def __init__(self, cfg: AnnotateSettings) -> None:
        self.cfg = cfg

    def color(self, track: Track) -> tuple[int, int, int]:
        key = track.role if track.is_person else "vehicle"
        return tuple(self.cfg.colors.get(key, self.cfg.colors["pending"]))  # type: ignore[return-value]

    @staticmethod
    def label(track: Track) -> str:
        if track.is_person:
            gid = track.global_id if track.global_id is not None else f"t{track.track_id}"
            return f"{track.role[0].upper()} #{gid}"
        plate = track.extra.get("plate")
        return f"{track.label} {plate}" if plate else f"{track.label} #{track.track_id}"

    def draw(self, image: np.ndarray, tracks: list[Track], header: str = "") -> np.ndarray:
        out = image.copy()
        t, fs = self.cfg.thickness, self.cfg.font_scale
        for tr in tracks:
            c = self.color(tr)
            x1, y1, x2, y2 = (int(v) for v in tr.xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), c, t)
            text = self.label(tr)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ty = max(th + 4, y1)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty), c, -1)
            cv2.putText(
                out, text, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA
            )
        if header:
            cv2.putText(
                out, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                out, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
        return out
