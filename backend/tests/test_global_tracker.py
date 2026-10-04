from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.global_tracker import GlobalTracker
from app.pipeline.reid import l2n

RNG = np.random.default_rng(0)
PROTOS = l2n(RNG.normal(size=(6, 512)))


def emb(person: int, noise: float = 0.15) -> np.ndarray:
    return l2n(PROTOS[person] + noise * RNG.normal(size=512) / np.sqrt(512) * 10)


@pytest.fixture
def gt(config):
    return GlobalTracker(config.settings.global_tracker)


def walk(gt, cam, tid, person, t0, t1, xy=None, step=0.2, end=True):
    """Feed one local track; xy is a callable t -> (x, y) or None. Returns last global id."""
    gid = None
    t = t0
    while t <= t1:
        gid = gt.observe(cam, tid, t, emb(person), 0.8, xy(t) if xy else None)
        t += step
    if end:
        gt.end(cam, tid)
    return gid


def test_same_person_across_cameras_keeps_id(gt):
    a = walk(gt, "A", 1, person=0, t0=0, t1=2)
    b = walk(gt, "B", 1, person=0, t0=5, t1=7)
    assert a is not None and a == b


def test_different_people_get_different_ids(gt):
    assert walk(gt, "A", 1, 0, 0, 2) != walk(gt, "B", 1, 1, 5, 7)


def test_pending_until_min_samples(gt, config):
    n = config.settings.global_tracker.min_samples
    ids = [gt.observe("A", 1, i * 0.1, emb(0), 0.8) for i in range(n)]
    assert ids[:-1] == [None] * (n - 1) and ids[-1] is not None


def test_two_live_tracks_in_one_camera_never_share_an_id(gt):
    for i in range(10):
        a = gt.observe("A", 1, i * 0.1, emb(0), 0.8)
        b = gt.observe("A", 2, i * 0.1, emb(0), 0.8)
    assert a != b


def test_concurrent_far_apart_sightings_are_different_people(gt):
    # identical appearance, seen at the same time 10 m apart in two calibrated cameras
    ids = {}
    for i in range(15):
        t = i * 0.1
        ids["A"] = gt.observe("A", 1, t, emb(2), 0.8, (0.0, 0.0))
        ids["B"] = gt.observe("B", 1, t, emb(2), 0.8, (10.0, 0.0))
    assert ids["A"] != ids["B"]


def test_concurrent_same_place_overrides_weak_appearance(gt):
    # different-looking crops (e.g. front vs back view) of one person at the same spot
    for i in range(15):
        t = i * 0.1
        a = gt.observe("A", 1, t, emb(3), 0.8, (1.0, 1.0))
        b = gt.observe("B", 1, t, l2n(0.5 * PROTOS[3] + 0.5 * PROTOS[4]), 0.8, (1.2, 1.1))
    assert a == b


def test_impossible_travel_speed_starts_new_identity(gt):
    a = walk(gt, "A", 1, 5, 0, 2, xy=lambda t: (0.0, 0.0))
    b = walk(gt, "B", 1, 5, 4, 6, xy=lambda t: (100.0, 0.0))  # 100 m in 2 s
    c = walk(gt, "C", 1, 5, 60, 62, xy=lambda t: (100.0, 0.0))  # 100 m in ~1 min: plausible
    assert a != b and c in (a, b)


def test_end_stale_closes_vanished_tracks(gt):
    gt.observe("A", 1, 0.0, emb(0), 0.8)
    gt.observe("A", 2, 0.0, emb(1), 0.8)
    assert gt.end_stale("A", {2}, ts=5.0, grace_s=2.0) == [1]
    assert ("A", 1) not in gt.sightings and ("A", 2) in gt.sightings
