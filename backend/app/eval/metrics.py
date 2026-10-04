"""Identity metrics (IDF1 / IDP / IDR, Ristani et al. 2016) for single- and multi-camera tracking.

Observations are keyed by (camera, frame). For every GT/pred trajectory pair we count the
observations where both exist and their boxes overlap with IoU >= threshold; a one-to-one
assignment of trajectories (Hungarian, maximizing matched observations) gives IDTP.
    IDF1 = 2 IDTP / (|GT obs| + |pred obs|),  IDP = IDTP / |pred obs|,  IDR = IDTP / |GT obs|
Multi-camera IDF1 uses global ids across all cameras, so a person switching id between
cameras is penalized exactly like an id switch within one camera.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = tuple[float, float, float, float]
# (camera, frame, identity, box)
Obs = tuple[str, int, Hashable, Box]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


@dataclass
class IdMetrics:
    idf1: float
    idp: float
    idr: float
    idtp: int
    idfp: int
    idfn: int
    gt_ids: int
    pred_ids: int
    gt_obs: int
    pred_obs: int

    def as_dict(self) -> dict:
        return asdict(self)


def id_metrics(gt: Iterable[Obs], pred: Iterable[Obs], iou_threshold: float = 0.5) -> IdMetrics:
    gt_by: dict[tuple[str, int], list[tuple[Hashable, Box]]] = defaultdict(list)
    pr_by: dict[tuple[str, int], list[tuple[Hashable, Box]]] = defaultdict(list)
    for cam, f, i, b in gt:
        gt_by[(cam, f)].append((i, b))
    for cam, f, i, b in pred:
        pr_by[(cam, f)].append((i, b))
    gt_ids = sorted({i for v in gt_by.values() for i, _ in v}, key=str)
    pr_ids = sorted({i for v in pr_by.values() for i, _ in v}, key=str)
    gi = {g: k for k, g in enumerate(gt_ids)}
    pi = {p: k for k, p in enumerate(pr_ids)}
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for key, gts in gt_by.items():
        prs = pr_by.get(key)
        if not prs:
            continue
        ious = iou_matrix(
            np.array([b for _, b in gts], float), np.array([b for _, b in prs], float)
        )
        for r, c in zip(*np.nonzero(ious >= iou_threshold), strict=True):
            counts[(gi[gts[r][0]], pi[prs[c][0]])] += 1
    n_gt = sum(len(v) for v in gt_by.values())
    n_pr = sum(len(v) for v in pr_by.values())
    idtp = 0
    if counts:
        m = np.zeros((len(gt_ids), len(pr_ids)))
        for (g, p), n in counts.items():
            m[g, p] = n
        rows, cols = linear_sum_assignment(-m)
        idtp = int(m[rows, cols].sum())
    return IdMetrics(
        idf1=2 * idtp / (n_gt + n_pr) if n_gt + n_pr else 0.0,
        idp=idtp / n_pr if n_pr else 0.0,
        idr=idtp / n_gt if n_gt else 0.0,
        idtp=idtp,
        idfp=n_pr - idtp,
        idfn=n_gt - idtp,
        gt_ids=len(gt_ids),
        pred_ids=len(pr_ids),
        gt_obs=n_gt,
        pred_obs=n_pr,
    )
