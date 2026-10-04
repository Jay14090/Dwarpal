from __future__ import annotations

import numpy as np
import pytest

from app.datasets.common import GroundTruth
from app.eval.mtmc import evaluate_mtmc, load_cameras
from app.pipeline.cache import TrackCacheWriter, cache_path
from app.pipeline.crosscam import ReidCacheWriter, reid_cache_path
from app.pipeline.frames import Track
from app.pipeline.reid import l2n

LABELS = ("person", "car")
RNG = np.random.default_rng(1)
PROTO = l2n(RNG.normal(size=(3, 512)))
# person -> list of (camera, first frame, last frame)
SCRIPT = {0: [("A", 0, 60), ("B", 90, 150)], 1: [("A", 0, 150)], 2: [("B", 0, 60)]}


def box(person: int, f: int) -> tuple[float, float, float, float]:
    x = 20 + 150 * person + (f % 40)
    return (x, 50.0, x + 40.0, 170.0)


@pytest.fixture
def dataset(tmp_path):
    ddir, cache = tmp_path / "processed" / "synth", tmp_path / "cache"
    for cam in ("A", "B"):
        rows, tw, rw = [], TrackCacheWriter(LABELS), ReidCacheWriter()
        local_id = {p: 10 * (p + 1) + (cam == "B") for p in SCRIPT}  # tracker ids differ per cam
        for f in range(160):
            tracks = []
            for p, spans in SCRIPT.items():
                if any(c == cam and a <= f <= b for c, a, b in spans):
                    rows.append([f, p, *box(p, f), "person"])
                    tracks.append(Track(local_id[p], box(p, f), 0.9, "person"))
                    if f % 5 == 0:
                        rw(f, local_id[p], 0.8, l2n(PROTO[p] + 0.05 * RNG.normal(size=512)))
            tw.add(f, tracks)
        GroundTruth("synth", cam, 30.0, 640, 360, 160, True, True, rows).save(
            ddir / "gt" / f"{cam}.json"
        )
        tw.save(cache_path(cache, cam), {"num_frames": 160})
        rw.save(reid_cache_path(cache, cam), {})
    return ddir, cache


def test_global_ids_across_cameras_score_perfectly_at_tracklet_level(dataset, config):
    ddir, cache = dataset
    cams = load_cameras(ddir, cache, ["A", "B"])
    res = evaluate_mtmc(cams, config.settings.reid, config.settings.global_tracker)
    assert res.single_camera.idf1 == 1.0
    assert res.multi_camera_tracklet.idf1 == 1.0
    assert res.global_ids == 3
    # pending frames (before min_samples Re-ID samples) cost a little online
    assert 0.8 < res.multi_camera_online.idf1 < 1.0


def test_appearance_only_threshold_too_high_splits_identity(dataset, config):
    ddir, cache = dataset
    cams = load_cameras(ddir, cache, ["A", "B"])
    strict = config.settings.global_tracker.model_copy(update={"match_threshold": 1.5})
    res = evaluate_mtmc(cams, config.settings.reid, strict)
    assert res.global_ids == 4  # person 0 gets a second id in camera B
    assert res.multi_camera_tracklet.idf1 < 1.0


def test_missing_reid_cache_is_reported(dataset, config):
    ddir, cache = dataset
    reid_cache_path(cache, "B").unlink()
    cams = load_cameras(ddir, cache, ["A", "B"])
    with pytest.raises(FileNotFoundError, match="reid"):
        evaluate_mtmc(cams, config.settings.reid, config.settings.global_tracker)
