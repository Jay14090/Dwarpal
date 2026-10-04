from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.identity import Gallery, GalleryPerson, IdentityEngine
from app.pipeline.reid import l2n

RNG = np.random.default_rng(3)
FACE = l2n(RNG.normal(size=(4, 512)))
BODY = l2n(RNG.normal(size=(4, 512)))


def near(v: np.ndarray, noise: float = 0.02) -> np.ndarray:
    return l2n(v + noise * RNG.normal(size=v.shape))


def gallery(*people: tuple[int, str, int]) -> Gallery:
    """people: (person_id, role, prototype index)."""
    return Gallery(
        {pid: GalleryPerson(pid, role) for pid, role, _ in people},
        {
            "face": [(pid, FACE[k]) for pid, _, k in people],
            "body": [(pid, BODY[k]) for pid, _, k in people],
        },
        version=1,
    )


@pytest.fixture
def cfg(config):
    return config.settings.identity


def test_unknown_after_n_unmatched_observations(cfg):
    eng = IdentityEngine(cfg, gallery((1, "resident", 0)))
    roles = [
        eng.observe(7, "body", near(BODY[3]), 0.9, t) for t in range(cfg.unknown_after_observations)
    ]
    assert roles[:-1] == ["pending"] * (cfg.unknown_after_observations - 1)
    assert roles[-1] == "unknown"


def test_face_match_labels_role_quickly(cfg):
    eng = IdentityEngine(cfg, gallery((1, "resident", 0), (2, "staff", 1)))
    assert eng.observe(7, "face", near(FACE[1]), 1.0, 0.0) == "staff"
    assert eng.states[7].person_id == 2


def test_body_evidence_accumulates_by_quality(cfg):
    eng = IdentityEngine(cfg, gallery((1, "resident", 0)))
    # body weight 0.5 x quality 0.5 = 0.25 per sample: needs 4 samples to reach accept_score 1.0
    roles = [eng.observe(7, "body", near(BODY[0]), 0.5, t) for t in range(4)]
    assert roles == ["pending", "pending", "pending", "resident"]


def test_low_quality_samples_are_ignored(cfg):
    eng = IdentityEngine(cfg, gallery((1, "resident", 0)))
    for t in range(50):
        eng.observe(7, "face", near(FACE[0]), cfg.min_quality / 2, t)
    assert eng.role(7) == "pending" and eng.states.get(7) is None


def test_no_flicker_between_similar_people(cfg):
    eng = IdentityEngine(cfg, gallery((1, "resident", 0), (2, "staff", 1)))
    eng.observe(7, "face", near(FACE[0]), 1.0, 0)
    eng.observe(7, "face", near(FACE[1]), 1.0, 1)  # one contrary sample is not enough
    assert eng.role(7) == "resident"
    for t in range(2, 6):
        eng.observe(7, "face", near(FACE[1]), 1.0, t)  # sustained contrary evidence switches
    assert eng.role(7) == "staff"


def test_enrollment_flips_unknown_to_resident_immediately(cfg):
    eng = IdentityEngine(cfg, Gallery())
    events = []
    eng.listeners.append(lambda st, prev, ts: events.append((st.global_id, prev, st.role_state)))
    for t in range(cfg.unknown_after_observations):
        eng.observe(9, "face", near(FACE[2]), 0.9, t)
    assert eng.role(9) == "unknown"
    eng.set_gallery(gallery((5, "resident", 2)), ts=100.0)  # person 9 is enrolled
    assert eng.role(9) == "resident"
    assert events == [(9, "pending", "unknown"), (9, "unknown", "resident")]


def test_deleting_a_person_resets_their_tracks(cfg):
    eng = IdentityEngine(cfg, gallery((1, "resident", 0)))
    eng.observe(7, "face", near(FACE[0]), 1.0, 0)
    eng.set_gallery(Gallery(), ts=1.0)
    assert eng.states[7].person_id is None and eng.role(7) in ("pending", "unknown")


def test_gallery_npz_roundtrip(tmp_path):
    g = gallery((1, "resident", 0), (2, "staff", 1))
    g.save_npz(tmp_path / "g.npz")
    h = Gallery.load_npz(tmp_path / "g.npz")
    assert h.people[2].role == "staff" and h.count("face") == 2
    assert h.best_match("body", BODY[1])[0] == 2
