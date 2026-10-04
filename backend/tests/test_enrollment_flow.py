from __future__ import annotations

import threading
import time

import numpy as np

from app.pipeline.engine import Engine
from app.pipeline.face import FaceSample
from app.pipeline.identity import Gallery, GalleryPerson
from app.pipeline.reid import ColorHistEncoder, l2n

from .test_pipeline import BrightBoxDetector, camera, engine_config, write_video

ALICE = l2n(np.random.default_rng(7).normal(size=512))


class FakeFaces:
    """Every person taller than 50 px shows Alice's face."""

    def faces_for(self, image, boxes):
        return [
            FaceSample(ALICE, 0.9, 0.99, (b[0], b[1], b[2], b[1] + 30))
            if b[3] - b[1] > 50
            else None
            for b in boxes
        ]


def wait_for(pred, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_unknown_person_enrolled_from_camera_becomes_resident(tmp_path, config):
    video = write_video(tmp_path / "v.mp4", n=60)
    cfg = engine_config(config, [camera(video)])
    s = cfg.settings
    # the synthetic person is 120 px tall in a 320x240 clip: scale the size gates to match
    ident = s.identity.model_copy(update={"unknown_after_observations": 3, "min_quality": 0.1})
    face = s.face.model_copy(update={"min_person_height_px": 50})
    cfg = cfg.model_copy(
        update={"settings": s.model_copy(update={"identity": ident, "face": face})}
    )

    roles: list[str] = []
    gallery_box: list[Gallery] = [Gallery()]
    eng = Engine(
        cfg,
        lambda c, j, m: roles.extend(t["role"] for t in m["tracks"]),
        detector=BrightBoxDetector(),
        encoder=ColorHistEncoder(),
        face_encoder=FakeFaces(),
        gallery_loader=lambda: gallery_box[0],
    )
    eng.start()
    try:
        assert wait_for(lambda: "unknown" in roles), roles[-5:]

        result: dict = {}
        done = threading.Event()
        eng.start_capture(
            "cam1", seconds=1.5, shots=5, on_done=lambda r: (result.update(r), done.set())
        )
        while not done.is_set():
            eng.tick()
            time.sleep(0.05)
        assert result["frames_with_person"] > 0
        assert 3 <= len(result["face"]) <= 5 and len(result["body"]) >= 3
        assert result["thumb_jpeg"][:2] == b"\xff\xd8"

        # what the API does after storing the person: reload the gallery
        gallery_box[0] = Gallery(
            {1: GalleryPerson(1, "resident", "Alice")},
            {"face": [(1, np.asarray(e, np.float32)) for e in result["face"]]},
            version=2,
        )
        roles.clear()
        eng.reload_gallery()
        assert wait_for(lambda: len(roles) > 3 and roles[-1] == "resident"), roles[-5:]
        # the 2 s clip loops, so the box may restart as a new track: every identity is Alice
        assert set(eng.stats()["identities"]["roles"]) == {"resident"}
    finally:
        eng.stop()


def test_capture_on_missing_camera_reports_error(config):
    eng = Engine(engine_config(config, []), lambda *a: None, encoder=ColorHistEncoder())
    out: list[dict] = []
    eng.start_capture("nope", 1, 3, out.append)
    assert "error" in out[0]
