"""Zone membership: a track is in a zone when its foot point lies inside the zone polygon."""

from __future__ import annotations

import cv2
import numpy as np

from app.core.config import Camera, Zone


def foot_norm(xyxy: tuple[float, float, float, float], w: int, h: int) -> tuple[float, float]:
    x1, _, x2, y2 = xyxy
    return ((x1 + x2) / 2 / w, y2 / h)


def in_zone(zone: Zone, point: tuple[float, float]) -> bool:
    poly = np.array(zone.polygon, np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def zones_at(camera: Camera, xyxy: tuple[float, float, float, float], w: int, h: int) -> list[str]:
    p = foot_norm(xyxy, w, h)
    return [z.name for z in camera.zones if in_zone(z, p)]
