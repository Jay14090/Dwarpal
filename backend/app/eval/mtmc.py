"""Offline multi-camera evaluation: replay cached tracks + Re-ID samples through the same
CrossCameraHook / GlobalTracker the live engine uses, then score against ground truth.

Cameras are interleaved frame by frame with ts = frame / fps, which is the synchronized
timeline of MTMC datasets (SmartSpaces frame indices are aligned across cameras).
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import GlobalTrackerSettings, ReidSettings
from app.datasets.common import Calibration, GroundTruth
from app.eval.metrics import IdMetrics, Obs, id_metrics
from app.pipeline.cache import TrackCache, cache_path
from app.pipeline.crosscam import CrossCameraHook, ReidCache, reid_cache_path
from app.pipeline.frames import Frame, FrameResult
from app.pipeline.global_tracker import GlobalTracker


@dataclass
class CameraData:
    camera_id: str
    gt: GroundTruth
    tracks: TrackCache
    reid: ReidCache | None
    calibration: Calibration | None


@dataclass
class MtmcResult:
    per_camera: dict[str, IdMetrics]
    single_camera: IdMetrics  # local track ids, GT ids namespaced per camera
    multi_camera_online: IdMetrics  # global ids as shown live (pending = own temporary id)
    multi_camera_tracklet: IdMetrics  # each tracklet relabeled with its final global id
    frames: tuple[int, int]
    global_ids: int
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": list(self.frames),
            "global_ids": self.global_ids,
            "per_camera": {k: v.as_dict() for k, v in self.per_camera.items()},
            "single_camera": self.single_camera.as_dict(),
            "multi_camera_online": self.multi_camera_online.as_dict(),
            "multi_camera_tracklet": self.multi_camera_tracklet.as_dict(),
            **self.extras,
        }


def load_cameras(dataset_dir: Path, cache_dir: Path, camera_ids: list[str]) -> list[CameraData]:
    out = []
    for cam in camera_ids:
        tp = cache_path(cache_dir, cam)
        if not tp.is_file():
            raise FileNotFoundError(f"{cam}: no track cache at {tp}; run `make index`")
        rp = reid_cache_path(cache_dir, cam)
        cp = dataset_dir / "calibration" / f"{cam}.json"
        out.append(
            CameraData(
                cam,
                GroundTruth.load(dataset_dir / "gt" / f"{cam}.json"),
                TrackCache(tp),
                ReidCache(rp) if rp.is_file() else None,
                Calibration.load(cp) if cp.is_file() else None,
            )
        )
    return out


def evaluate_mtmc(
    cams: list[CameraData],
    reid_cfg: ReidSettings,
    gt_cfg: GlobalTrackerSettings,
    start_frame: int = 0,
    end_frame: int | None = None,
    use_calibration: bool = True,
    track_buffer: int = 30,
) -> MtmcResult:
    if any(c.reid is None for c in cams):
        missing = [c.camera_id for c in cams if c.reid is None]
        raise FileNotFoundError(f"no reid cache for {missing}; run `make index` with Re-ID enabled")
    fps = cams[0].gt.fps
    n = min(min(c.tracks.num_frames, c.gt.num_frames) for c in cams)
    end = n if end_frame is None else min(end_frame, n)
    tracker = GlobalTracker(gt_cfg, reid_cfg.max_samples)
    hooks = {
        c.camera_id: CrossCameraHook(
            c.camera_id, reid_cfg, tracker, cache=c.reid,
            calibration=c.calibration if use_calibration else None,
            grace_s=track_buffer / fps + 0.5,
        )
        for c in cams
    }  # fmt: skip
    dummy = np.zeros((1, 1, 3), np.uint8)
    online: list[Obs] = []
    local: list[Obs] = []
    tracklet_obs: list[tuple[str, int, tuple[str, int], tuple[float, float, float, float]]] = []
    final_gid: dict[tuple[str, int], int] = {}
    for f in range(start_frame, end):
        for c in cams:
            tracks = c.tracks.tracks_at(f)
            hooks[c.camera_id](FrameResult(Frame(c.camera_id, f, f / fps, dummy), tracks))
            for t in tracks:
                if not t.is_person:
                    continue
                key = (c.camera_id, t.track_id)
                gid: Hashable = t.global_id if t.global_id is not None else ("pending", *key)
                if t.global_id is not None:
                    final_gid[key] = t.global_id
                online.append((c.camera_id, f, gid, t.xyxy))
                local.append((c.camera_id, f, key, t.xyxy))
                tracklet_obs.append((c.camera_id, f, key, t.xyxy))
    tracklet: list[Obs] = [
        (cam, f, final_gid.get(key, ("unassigned", *key)), box) for cam, f, key, box in tracklet_obs
    ]
    gt_global: list[Obs] = []
    gt_local: list[Obs] = []
    for c in cams:
        for fr, gid, x1, y1, x2, y2, cls_ in c.gt.rows:
            if cls_ == "person" and start_frame <= fr < end:
                gt_global.append((c.camera_id, fr, gid, (x1, y1, x2, y2)))
                gt_local.append((c.camera_id, fr, (c.camera_id, gid), (x1, y1, x2, y2)))
    per_camera = {
        c.camera_id: id_metrics(
            [o for o in gt_local if o[0] == c.camera_id], [o for o in local if o[0] == c.camera_id]
        )
        for c in cams
    }
    return MtmcResult(
        per_camera=per_camera,
        single_camera=id_metrics(gt_local, local),
        multi_camera_online=id_metrics(gt_global, online),
        multi_camera_tracklet=id_metrics(gt_global, tracklet),
        frames=(start_frame, end),
        global_ids=len(tracker.identities),
    )
