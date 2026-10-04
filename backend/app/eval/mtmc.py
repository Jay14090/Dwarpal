"""Offline multi-camera evaluation: replay cached tracks + Re-ID samples through the same
CrossCameraHook / GlobalTracker the live engine uses, then score against ground truth.

Cameras are interleaved frame by frame with ts = frame / fps, which is the synchronized
timeline of MTMC datasets (SmartSpaces frame indices are aligned across cameras).
"""

from __future__ import annotations

from collections.abc import Callable
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
from app.pipeline.worker import FrameHook


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


@dataclass
class ReplayObs:
    camera_id: str
    frame: int
    track: tuple[str, int]  # (camera, local track id)
    global_id: int | None
    role: str
    box: tuple[float, float, float, float]


HookFactory = Callable[[CameraData], list[FrameHook]]


def replay(
    cams: list[CameraData],
    reid_cfg: ReidSettings,
    gt_cfg: GlobalTrackerSettings,
    start_frame: int = 0,
    end_frame: int | None = None,
    use_calibration: bool = True,
    track_buffer: int = 30,
    extra_hooks: HookFactory | None = None,
) -> tuple[list[ReplayObs], GlobalTracker, tuple[int, int]]:
    """Run cached tracks of all cameras through the live hooks in synchronized frame order."""
    if any(c.reid is None for c in cams):
        missing = [c.camera_id for c in cams if c.reid is None]
        raise FileNotFoundError(f"no reid cache for {missing}; run `make index` with Re-ID enabled")
    fps = cams[0].gt.fps
    n = min(min(c.tracks.num_frames, c.gt.num_frames) for c in cams)
    end = n if end_frame is None else min(end_frame, n)
    tracker = GlobalTracker(gt_cfg, reid_cfg.max_samples)
    hooks: dict[str, list[FrameHook]] = {}
    for c in cams:
        hooks[c.camera_id] = [
            CrossCameraHook(
                c.camera_id, reid_cfg, tracker, cache=c.reid,
                calibration=c.calibration if use_calibration else None,
                grace_s=track_buffer / fps + 0.5,
            ),
            *(extra_hooks(c) if extra_hooks else []),
        ]  # fmt: skip
    dummy = np.zeros((1, 1, 3), np.uint8)
    obs: list[ReplayObs] = []
    for f in range(start_frame, end):
        for c in cams:
            tracks = c.tracks.tracks_at(f)
            result = FrameResult(Frame(c.camera_id, f, f / fps, dummy), tracks)
            for hook in hooks[c.camera_id]:
                hook(result)
            obs.extend(
                ReplayObs(c.camera_id, f, (c.camera_id, t.track_id), t.global_id, t.role, t.xyxy)
                for t in tracks
                if t.is_person
            )
    return obs, tracker, (start_frame, end)


def evaluate_mtmc(
    cams: list[CameraData],
    reid_cfg: ReidSettings,
    gt_cfg: GlobalTrackerSettings,
    start_frame: int = 0,
    end_frame: int | None = None,
    use_calibration: bool = True,
    track_buffer: int = 30,
) -> MtmcResult:
    obs, tracker, (start, end) = replay(
        cams, reid_cfg, gt_cfg, start_frame, end_frame, use_calibration, track_buffer
    )
    final_gid: dict[tuple[str, int], int] = {
        o.track: o.global_id for o in obs if o.global_id is not None
    }
    online: list[Obs] = [
        (
            o.camera_id,
            o.frame,
            o.global_id if o.global_id is not None else ("pending", *o.track),
            o.box,
        )
        for o in obs
    ]
    local: list[Obs] = [(o.camera_id, o.frame, o.track, o.box) for o in obs]
    tracklet: list[Obs] = [
        (o.camera_id, o.frame, final_gid.get(o.track, ("unassigned", *o.track)), o.box) for o in obs
    ]
    gt_global: list[Obs] = []
    gt_local: list[Obs] = []
    for c in cams:
        for fr, gid, x1, y1, x2, y2, cls_ in c.gt.rows:
            if cls_ == "person" and start <= fr < end:
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
        frames=(start, end),
        global_ids=len(tracker.identities),
    )
