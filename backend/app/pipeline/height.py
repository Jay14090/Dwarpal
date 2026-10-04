"""Person height from a calibrated camera (world z up, metres, ground plane z = 0).

Foot point (bottom-centre of the box) -> ground position (x, y) via the homography. The head
height h is where the projection of (x, y, h) reaches the top of the box: with P's image rows
p_u, p_v, p_w, solve (p_v - v_top * p_w) . [x, y, h, 1] = 0 for h. Per track the median of
valid frames is reported, with a robust spread (1.4826 * MAD) as the uncertainty.
Boxes cut by the image border are skipped (their top or bottom is not the head or feet).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.datasets.common import Calibration

MIN_CM, MAX_CM = 120.0, 220.0  # plausible adult range; outside means a bad box or occlusion


def height_m(cal: Calibration, xyxy: tuple[float, float, float, float]) -> float | None:
    x1, y1, x2, y2 = xyxy
    if y1 <= 2 or y2 >= cal.height - 2 or x1 <= 2 or x2 >= cal.width - 2:
        return None
    gx, gy = cal.image_to_ground(np.array([[(x1 + x2) / 2, y2]]))[0]
    pv, pw = cal.P[1], cal.P[2]
    a = pv - y1 * pw
    if abs(a[2]) < 1e-9:
        return None
    h = -(a[0] * gx + a[1] * gy + a[3]) / a[2]
    return float(h)


@dataclass
class HeightEstimate:
    samples: list[float] = field(default_factory=list)

    def add(self, cal: Calibration | None, xyxy: tuple[float, float, float, float]) -> None:
        if cal is None:
            return
        h = height_m(cal, xyxy)
        if h is not None and MIN_CM <= h * 100 <= MAX_CM:
            self.samples.append(h * 100)

    def result(self, min_samples: int = 3) -> tuple[float | None, float | None]:
        if len(self.samples) < min_samples:
            return None, None
        a = np.array(self.samples)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med))) * 1.4826
        return round(med, 1), round(max(mad, 2.0), 1)  # never claim better than +/- 2 cm
