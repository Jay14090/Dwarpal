"""Role-label accuracy and unknown-alert precision/recall on the held-out window (P4 DoD).

Replays frames after the simulated-enrollment window through the live hooks (global IDs,
faces if cached, IdentityEngine with the simulated gallery) and matches predicted person boxes
to GT boxes per frame (Hungarian, IoU >= 0.5).

Role-label accuracy: over matched observations whose predicted role is decided
    (resident/staff/unknown); `coverage` is the share of matched observations that are decided.
Unknown alerts: one alert per global identity the first time it turns `unknown` (the rules
    engine dedups the same way). An alert is correct when the GT person behind that identity
    (majority of its matched observations) is not enrolled. Recall is over unenrolled GT people
    seen after the window.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.core.config import FaceSettings, GlobalTrackerSettings, IdentitySettings, ReidSettings
from app.eval.metrics import iou_matrix
from app.eval.mtmc import CameraData, replay
from app.pipeline.identity import Gallery, IdentityEngine
from app.pipeline.person_hooks import FaceHook, IdentityHook, SampleCache

ROLES = ("resident", "staff", "unknown")


@dataclass
class IdentityResult:
    frames: tuple[int, int]
    matched_obs: int
    decided_obs: int
    correct_obs: int
    role_accuracy: float
    coverage: float
    confusion: dict[str, dict[str, int]]  # true role -> predicted role -> count
    alerts: int
    alerts_correct: int
    unknown_precision: float
    unknown_people: int
    unknown_people_alerted: int
    unknown_recall: float
    false_unknown_people: list[int] = field(
        default_factory=list
    )  # enrolled GT ids alerted as unknown

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def match_frame(
    gt: list[tuple[int, tuple]], pred: list[int], pred_boxes: list[tuple], thr: float
) -> dict[int, int]:
    """pred index -> GT id for one camera frame (one-to-one, IoU >= thr)."""
    if not gt or not pred:
        return {}
    ious = iou_matrix(np.array(pred_boxes, float), np.array([b for _, b in gt], float))
    rows, cols = linear_sum_assignment(-ious)
    return {pred[r]: gt[c][0] for r, c in zip(rows, cols, strict=True) if ious[r, c] >= thr}


def evaluate_identity(
    cams: list[CameraData],
    reid_cfg: ReidSettings,
    gt_cfg: GlobalTrackerSettings,
    id_cfg: IdentitySettings,
    face_cfg: FaceSettings,
    gallery: Gallery,
    enrolled_roles: dict[int, str],  # GT id -> resident/staff
    start_frame: int,
    face_caches: dict[str, SampleCache] | None = None,
    end_frame: int | None = None,
    track_buffer: int = 30,
    iou_threshold: float = 0.5,
) -> IdentityResult:
    identity = IdentityEngine(id_cfg, gallery)
    alerts: list[int] = []
    identity.listeners.append(
        lambda st, prev, ts: (
            alerts.append(st.global_id)
            if st.role_state == "unknown" and prev == "pending"
            else None
        )
    )

    def hooks(c: CameraData) -> list:
        extra: list = []
        if face_caches and c.camera_id in face_caches:
            extra.append(FaceHook(c.camera_id, face_cfg, cache=face_caches[c.camera_id]))
        return [*extra, IdentityHook(identity)]

    obs, _, (start, end) = replay(
        cams, reid_cfg, gt_cfg, start_frame, end_frame, track_buffer=track_buffer, extra_hooks=hooks
    )

    gt_by: dict[tuple[str, int], list[tuple[int, tuple]]] = defaultdict(list)
    unknown_people: set[int] = set()
    for c in cams:
        for f, gid, x1, y1, x2, y2, cls_ in c.gt.rows:
            if cls_ == "person" and start <= f < end:
                gt_by[(c.camera_id, f)].append((gid, (x1, y1, x2, y2)))
                if gid not in enrolled_roles:
                    unknown_people.add(gid)
    pred_by: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, o in enumerate(obs):
        pred_by[(o.camera_id, o.frame)].append(i)

    confusion: dict[str, Counter[str]] = {r: Counter() for r in ROLES}
    gid_votes: dict[int, Counter[int]] = defaultdict(Counter)  # predicted global id -> GT ids
    matched = decided = correct = 0
    for key, idxs in pred_by.items():
        m = match_frame(gt_by.get(key, []), idxs, [obs[i].box for i in idxs], iou_threshold)
        for i, gt_id in m.items():
            o = obs[i]
            true_role = enrolled_roles.get(gt_id, "unknown")
            matched += 1
            confusion[true_role][o.role] += 1
            if o.role in ROLES:
                decided += 1
                correct += o.role == true_role
            if o.global_id is not None:
                gid_votes[o.global_id][gt_id] += 1

    alerted_gt: dict[int, int] = {}
    for g in alerts:
        votes = gid_votes.get(g)
        if votes:
            alerted_gt[g] = votes.most_common(1)[0][0]
    alerts_correct = sum(
        1 for g in alerts if g in alerted_gt and alerted_gt[g] not in enrolled_roles
    )
    covered = {
        alerted_gt[g] for g in alerts if g in alerted_gt and alerted_gt[g] not in enrolled_roles
    }
    false_unknown = sorted(
        {alerted_gt[g] for g in alerts if g in alerted_gt and alerted_gt[g] in enrolled_roles}
    )
    return IdentityResult(
        frames=(start, end),
        matched_obs=matched,
        decided_obs=decided,
        correct_obs=correct,
        role_accuracy=correct / decided if decided else 0.0,
        coverage=decided / matched if matched else 0.0,
        confusion={k: dict(v) for k, v in confusion.items()},
        alerts=len(alerts),
        alerts_correct=alerts_correct,
        unknown_precision=alerts_correct / len(alerts) if alerts else 0.0,
        unknown_people=len(unknown_people),
        unknown_people_alerted=len(covered),
        unknown_recall=len(covered) / len(unknown_people) if unknown_people else 0.0,
        false_unknown_people=false_unknown,
    )
