"""NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2024 layout) parsing.

Formats were read from the dataset README and real files of `MTMC_Tracking_2024/test/scene_071`:
- scene/ground_truth.txt: `camera_id obj_id frame_id xmin ymin width height xworld yworld`
  (whitespace separated, frame 0-based, obj_id consistent across cameras)
- scene/camera_XXXX/calibration.json: {"camera projection matrix": 3x4, "homography matrix": 3x3, ...}
- scene/calibration_2025_format.json: {"sensors": [{"id": "Camera_0635", "intrinsicMatrix",
  "extrinsicMatrix" (3x4 [R|t]), "cameraMatrix" (3x4), "homography", "attributes": [...]}]}
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from app.datasets.common import Calibration

REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
CAMERA_DIR_RE = re.compile(r"camera_(\d+)$")


def camera_slug(camera_num: int) -> str:
    return f"ss_{camera_num:04d}"


def iter_ground_truth(path: Path) -> Iterator[tuple[int, int, int, float, float, float, float]]:
    """Yield (camera_id, obj_id, frame, x1, y1, x2, y2) from a 2024 ground_truth.txt."""
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 7:
                raise ValueError(f"{path.name}:{lineno}: expected >= 7 fields, got {len(parts)}")
            cam, obj, frame = int(parts[0]), int(parts[1]), int(parts[2])
            x, y, w, h = (float(v) for v in parts[3:7])
            yield cam, obj, frame, x, y, x + w, y + h


def camera_stats(gt_path: Path) -> dict[int, dict[str, int]]:
    """Per camera: number of boxes and distinct identities (used to pick busy cameras)."""
    boxes: Counter[int] = Counter()
    ids: dict[int, set[int]] = defaultdict(set)
    for cam, obj, *_ in iter_ground_truth(gt_path):
        boxes[cam] += 1
        ids[cam].add(obj)
    return {c: {"boxes": boxes[c], "identities": len(ids[c])} for c in boxes}


def identity_sets(gt_path: Path) -> dict[int, set[int]]:
    ids: dict[int, set[int]] = defaultdict(set)
    for cam, obj, *_ in iter_ground_truth(gt_path):
        ids[cam].add(obj)
    return dict(ids)


def pick_cameras(
    ids_per_camera: dict[int, set[int]], n: int, exclude: set[int] | None = None
) -> list[int]:
    """Greedy pick: start from the busiest camera, then add the camera sharing the most
    identities with those already picked (ties broken by size). Maximizes cross-camera overlap,
    which is what global-ID evaluation needs."""
    pool = {c: s for c, s in ids_per_camera.items() if c not in (exclude or set())}
    if not pool:
        return []
    chosen = [max(pool, key=lambda c: (len(pool[c]), -c))]
    seen = set(pool[chosen[0]])
    while len(chosen) < min(n, len(pool)):
        rest = [c for c in pool if c not in chosen]
        best = max(rest, key=lambda c: (len(pool[c] & seen), len(pool[c]), -c))
        chosen.append(best)
        seen |= pool[best]
    return chosen


def load_calibration(scene_dir: Path, camera_num: int, width: int, height: int) -> Calibration:
    """Prefer the 2025-format scene file (has K, R, t); fall back to the per-camera file."""
    scene_file = scene_dir / "calibration_2025_format.json"
    if scene_file.is_file():
        data = json.loads(scene_file.read_text())
        for sensor in data.get("sensors", []):
            m = re.search(r"(\d+)$", str(sensor.get("id", "")))
            if m and int(m.group(1)) == camera_num:
                ext = np.asarray(sensor["extrinsicMatrix"], dtype=float)
                return Calibration(
                    camera=camera_slug(camera_num),
                    width=width,
                    height=height,
                    P=np.asarray(sensor["cameraMatrix"], dtype=float),
                    K=np.asarray(sensor["intrinsicMatrix"], dtype=float),
                    R=ext[:, :3],
                    t=ext[:, 3],
                    source="smartspaces/calibration_2025_format.json",
                )
    cam_file = scene_dir / f"camera_{camera_num:04d}" / "calibration.json"
    data = json.loads(cam_file.read_text())
    return Calibration(
        camera=camera_slug(camera_num),
        width=width,
        height=height,
        P=np.asarray(data["camera projection matrix"], dtype=float),
        source="smartspaces/camera/calibration.json",
    )
