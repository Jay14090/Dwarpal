from __future__ import annotations

import pytest

from app.eval.metrics import id_metrics

B1, B2 = (0, 0, 10, 10), (100, 100, 110, 110)


def traj(cam, ident, frames, box):
    return [(cam, f, ident, box) for f in frames]


def test_perfect_tracking_is_one():
    gt = traj("c", 1, range(10), B1) + traj("c", 2, range(10), B2)
    pred = traj("c", "a", range(10), B1) + traj("c", "b", range(10), B2)
    m = id_metrics(gt, pred)
    assert m.idf1 == m.idp == m.idr == 1.0 and m.idtp == 20


def test_id_switch_halfway():
    gt = traj("c", 1, range(10), B1)
    pred = traj("c", "a", range(5), B1) + traj("c", "b", range(5, 10), B1)
    m = id_metrics(gt, pred)
    assert m.idtp == 5 and m.idf1 == pytest.approx(0.5)


def test_cross_camera_id_change_is_penalized():
    gt = traj("c1", 7, range(10), B1) + traj("c2", 7, range(10, 20), B1)
    same = traj("c1", "x", range(10), B1) + traj("c2", "x", range(10, 20), B1)
    split = traj("c1", "x", range(10), B1) + traj("c2", "y", range(10, 20), B1)
    assert id_metrics(gt, same).idf1 == 1.0
    assert id_metrics(gt, split).idf1 == pytest.approx(0.5)


def test_false_positives_and_misses():
    gt = traj("c", 1, range(10), B1)
    pred = traj("c", "a", range(5), B1) + traj("c", "fp", range(10), B2)
    m = id_metrics(gt, pred)
    assert (m.idtp, m.idfn, m.idfp) == (5, 5, 10)
    assert m.idf1 == pytest.approx(2 * 5 / (10 + 15))


def test_iou_threshold_applies():
    gt = traj("c", 1, range(4), (0, 0, 10, 10))
    pred = traj("c", "a", range(4), (5, 0, 15, 10))  # IoU 1/3
    assert id_metrics(gt, pred).idtp == 0
    assert id_metrics(gt, pred, iou_threshold=0.3).idtp == 4


def test_empty_inputs():
    assert id_metrics([], []).idf1 == 0.0
