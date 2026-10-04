"""Clothing colours of person crops: dominant colour of the upper and lower body in HSV.

The crop is split into torso (15-50 % of the height) and legs (55-90 %), trimmed to the central
50 % of the width to cut background, clustered with k-means (k=3) in HSV, and the largest
cluster's centre is mapped to one of 11 colour names. Per track, names are voted over the best
crops weighted by crop quality.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

COLORS = (
    "black",
    "white",
    "gray",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
)


def hsv_to_name(h: float, s: float, v: float) -> str:
    """OpenCV HSV (h 0-180, s/v 0-255) -> colour name."""
    s, v = s / 255.0, v / 255.0
    if v < 0.22:
        return "black"
    if s < 0.18:
        return "white" if v > 0.78 else "gray" if v > 0.3 else "black"
    deg = h * 2.0
    if deg < 12 or deg >= 340:
        return "pink" if s < 0.45 and v > 0.6 else "red"
    if deg < 40:
        return "brown" if v < 0.6 else "orange"
    if deg < 70:
        return "brown" if v < 0.45 else "yellow"
    if deg < 165:
        return "green"
    if deg < 255:
        return "blue"
    if deg < 290:
        return "purple"
    return "pink"


def _features(hsv: np.ndarray) -> np.ndarray:
    # hue is circular: (cos, sin) of hue scaled by saturation, plus s and v
    ang = hsv[:, 0] * (2 * np.pi / 180)
    return np.stack([np.cos(ang) * hsv[:, 1], np.sin(ang) * hsv[:, 1], hsv[:, 1], hsv[:, 2]], 1)


def _hsv(region: np.ndarray, size: int = 24) -> np.ndarray:
    small = cv2.resize(region, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)


def dominant_color(
    region: np.ndarray,
    k: int = 3,
    background: np.ndarray | None = None,
    bg_dist: float = 45.0,
    min_keep: float = 0.2,
) -> tuple[str, float]:
    """(name, share of pixels in the dominant cluster).

    `background`: HSV pixels of the scene next to the person (box side strips). Region pixels
    closer than `bg_dist` (feature space) to a background cluster are dropped first, unless
    that would leave less than `min_keep` of the region (then the region is used as is).
    """
    if region.size == 0 or min(region.shape[:2]) < 4:
        return "unknown", 0.0
    hsv = _hsv(region)
    feats = _features(hsv)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    if background is not None and len(background) >= 8:
        _, _, centers = cv2.kmeans(
            _features(background), 2, None, criteria, 2, cv2.KMEANS_PP_CENTERS
        )
        d = np.min(np.linalg.norm(feats[:, None, :] - centers[None], axis=2), axis=1)
        keep = d >= bg_dist
        if keep.mean() >= min_keep and keep.sum() >= k:
            hsv, feats = hsv[keep], feats[keep]
    _, labels, _ = cv2.kmeans(feats, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    best = int(np.argmax(counts))
    members = hsv[labels.ravel() == best]
    hue = (
        float(
            np.degrees(
                np.arctan2(
                    np.sin(members[:, 0] * np.pi / 90).mean(),
                    np.cos(members[:, 0] * np.pi / 90).mean(),
                )
            )
            % 360
        )
        / 2
    )
    return hsv_to_name(
        hue, float(np.median(members[:, 1])), float(np.median(members[:, 2]))
    ), float(counts[best] / counts.sum())


def body_regions(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = crop.shape[:2]
    x1, x2 = int(0.25 * w), int(0.75 * w)
    return crop[int(0.15 * h) : int(0.5 * h), x1:x2], crop[int(0.55 * h) : int(0.9 * h), x1:x2]


def side_strips(crop: np.ndarray, y1: float, y2: float, frac: float = 0.12) -> np.ndarray:
    """HSV pixels of the left/right edges of the box between rows y1..y2 (mostly background)."""
    h, w = crop.shape[:2]
    b = max(1, int(frac * w))
    rows = crop[int(y1 * h) : int(y2 * h)]
    if rows.size == 0 or w < 8:
        return np.zeros((0, 3), np.float32)
    return np.concatenate([_hsv(rows[:, :b], 8), _hsv(rows[:, w - b :], 8)])


def clothing_colors(
    crop: np.ndarray, background_filter: bool = True
) -> tuple[tuple[str, float], tuple[str, float]]:
    """Upper/lower dominant colours. The legs are thin, so floor shows between and beside them:
    the lower region drops pixels that match the box's side strips (P6 spot-check: this fixed
    "yellow pants" on a wooden floor). The torso fills the box, so the upper region is used as is
    (filtering it removed real clothing pixels)."""
    upper, lower = body_regions(crop)
    bl = side_strips(crop, 0.55, 0.95) if background_filter else None
    return dominant_color(upper), dominant_color(lower, background=bl)


@dataclass
class ColorVote:
    upper: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    lower: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, crop: np.ndarray, quality: float) -> None:
        (u, us), (lo, ls) = clothing_colors(crop)
        if u != "unknown":
            self.upper[u] += quality * us
        if lo != "unknown":
            self.lower[lo] += quality * ls

    @staticmethod
    def _winner(votes: dict[str, float]) -> tuple[str | None, float]:
        if not votes:
            return None, 0.0
        name = max(votes, key=votes.get)  # type: ignore[arg-type]
        return name, votes[name] / sum(votes.values())

    def result(self) -> tuple[str | None, str | None, float, float]:
        (u, uc), (lo, lc) = self._winner(self.upper), self._winner(self.lower)
        return u, lo, uc, lc
