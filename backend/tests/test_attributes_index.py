from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import Camera, Zone
from app.pipeline.attributes import ColorVote, clothing_colors, hsv_to_name
from app.pipeline.frames import Frame, FrameResult, Track
from app.pipeline.height import HeightEstimate, height_m
from app.pipeline.indexer import TrackIndexer
from app.zones import in_zone, zones_at

from .test_datasets import real_calibration


def bgr_name(bgr) -> str:
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    return hsv_to_name(float(h), float(s), float(v))


@pytest.mark.parametrize(
    ("bgr", "name"),
    [((10, 10, 10), "black"), ((245, 245, 245), "white"), ((128, 128, 128), "gray"), ((30, 30, 220), "red"),
     ((0, 140, 255), "orange"), ((0, 230, 230), "yellow"), ((40, 180, 40), "green"), ((200, 80, 20), "blue"),
     ((160, 40, 120), "purple"), ((180, 105, 255), "pink"), ((30, 70, 120), "brown")],
)  # fmt: skip
def test_named_colors(bgr, name):
    assert bgr_name(bgr) == name


def person_crop(upper=(30, 30, 220), lower=(200, 80, 20), h=240, w=100):
    img = np.full((h, w, 3), 90, np.uint8)  # gray background
    img[int(0.1 * h) : int(0.52 * h), int(0.2 * w) : int(0.8 * w)] = upper
    img[int(0.52 * h) : int(0.95 * h), int(0.25 * w) : int(0.75 * w)] = lower
    return img


def test_clothing_colors_split_upper_and_lower():
    (u, us), (lo, ls) = clothing_colors(person_crop())
    assert (u, lo) == ("red", "blue") and us > 0.6 and ls > 0.6


def test_color_vote_weights_by_quality():
    v = ColorVote()
    v.add(person_crop(upper=(40, 180, 40)), 0.9)  # green, good crop
    v.add(person_crop(upper=(30, 30, 220)), 0.2)  # red, poor crop
    upper, lower, uc, _ = v.result()
    assert upper == "green" and lower == "blue" and uc > 0.7


def project_box(cal, x, y, height_m_, half_w_px=30):
    foot, head = cal.project(np.array([[x, y, 0.0], [x, y, height_m_]]))
    return (foot[0] - half_w_px, head[1], foot[0] + half_w_px, foot[1])


def test_height_from_real_calibration():
    cal = real_calibration()  # SmartSpaces scene_071 camera 0635 (1920x1080)
    # pick ground points that land inside the image
    world = cal.image_to_ground(np.array([[960.0, 900.0], [700.0, 1000.0], [1300.0, 850.0]]))
    for x, y in world:
        for h in (1.55, 1.75, 1.92):
            box = project_box(cal, x, y, h)
            assert height_m(cal, box) == pytest.approx(h, abs=1e-6)


def test_height_estimate_median_and_skips_truncated():
    cal = real_calibration()
    x, y = cal.image_to_ground(np.array([[960.0, 900.0]]))[0]
    est = HeightEstimate()
    for h in (1.70, 1.72, 1.74, 1.76, 2.9):  # last is an outlier (out of range, dropped)
        est.add(cal, project_box(cal, x, y, h))
    est.add(cal, (0.0, 100.0, 50.0, 900.0))  # touches the left border: skipped
    med, err = est.result()
    assert med == pytest.approx(173.0, abs=0.6) and 2.0 <= err < 5
    assert HeightEstimate().result() == (None, None)


def test_zones():
    z = Zone(name="gate", polygon=[(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)])
    assert in_zone(z, (0.5, 0.75)) and not in_zone(z, (0.5, 0.25))
    cam = Camera(id="c", name="c", source_type="file", source_uri="x", zones=[z])
    assert zones_at(cam, (100, 100, 140, 400), 640, 480) == ["gate"]  # foot y 400/480 = 0.83
    assert zones_at(cam, (100, 10, 140, 100), 640, 480) == []


class FakeClip:
    def embed_images(self, crops):
        return np.tile(np.eye(512, dtype=np.float32)[0], (len(crops), 1))


def test_track_indexer_finishes_tracks_with_attributes(config):
    zone = Zone(name="lobby", polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    cam = Camera(id="c", name="c", source_type="file", source_uri="x", zones=[zone])
    out = []
    ix = TrackIndexer(cam, config.settings.index, out.append, clip=FakeClip(), grace_frames=10)
    frame_img = np.full((480, 640, 3), 90, np.uint8)
    frame_img[100:400, 280:380] = person_crop(h=300, w=100)
    for f in range(60):  # 2 s at 30 fps, then gone
        tracks = (
            [Track(4, (280.0, 100.0, 380.0, 400.0), 0.9, "person", global_id=12, role="resident")]
            if f < 40
            else []
        )
        if f < 40:
            tracks[0].extra["body_sample"] = (0.8, np.eye(512, dtype=np.float32)[1])
        ix(FrameResult(Frame("c", f, 1000 + f / 30, frame_img), tracks))
    assert len(out) == 1
    t = out[0]
    assert (t.global_id, t.role, t.local_track_id) == (12, "resident", 4)
    assert (t.upper_color, t.lower_color) == ("red", "blue")
    assert t.zones == ["lobby"] and t.start_frame == 0 and t.end_frame == 39
    assert t.clip is not None and t.body is not None and t.thumb_jpeg[:2] == b"\xff\xd8"
    assert t.height_cm is None  # no calibration


def test_short_tracks_are_not_indexed(config):
    cam = Camera(id="c", name="c", source_type="file", source_uri="x")
    out = []
    ix = TrackIndexer(cam, config.settings.index, out.append, grace_frames=2)
    img = np.full((480, 640, 3), 90, np.uint8)
    for f in range(10):  # 0.17 s
        ix(
            FrameResult(
                Frame("c", f, f / 30, img),
                [Track(1, (10, 10, 60, 200), 0.9, "person")] if f < 5 else [],
            )
        )
    assert out == []


@pytest.mark.db
def test_track_store_writes_rows_and_thumbnail(migrated_db_url, config, tmp_path):
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.db.models import GlobalIdentity, TrackEmbedding
    from app.db.models import Track as TrackRow
    from app.db.sync import sync_cameras
    from app.db.track_store import TrackStore, next_global_id
    from app.pipeline.indexer import DbWriter

    db = create_engine(migrated_db_url)
    writer = DbWriter(db)
    writer.submit(lambda s: sync_cameras(s, config.cameras))
    stored = []
    store = TrackStore(writer, tmp_path, on_stored=lambda tid, ft: stored.append(tid))
    cam = config.cameras.cameras[0]
    ix = TrackIndexer(cam, config.settings.index, store.store, clip=FakeClip(), grace_frames=5)
    img = np.full((480, 640, 3), 90, np.uint8)
    img[100:400, 280:380] = person_crop(h=300, w=100)
    with Session(db) as s:
        gid = next_global_id(s)
    store.identity_created(gid, 1000.0)
    for f in range(50):
        tr = (
            [Track(1, (280.0, 100.0, 380.0, 400.0), 0.9, "person", global_id=gid, role="unknown")]
            if f < 40
            else []
        )
        ix(FrameResult(Frame(cam.id, f, 1000 + f / 30, img), tr))
    store.identity_changed(gid, "unknown", None, 1002.0)
    deadline = time.monotonic() + 5
    while not stored and time.monotonic() < deadline:
        time.sleep(0.05)
    writer.flush()
    writer.close()
    with Session(db) as s:
        row = s.get(TrackRow, stored[0])
        assert (row.upper_color, row.lower_color, row.role, row.global_id) == (
            "red",
            "blue",
            "unknown",
            gid,
        )
        assert row.thumb_path and Path(row.thumb_path).read_bytes()[:2] == b"\xff\xd8"
        emb = s.get(TrackEmbedding, row.id)
        assert emb.clip is not None and len(emb.clip) == 512
        ident = s.get(GlobalIdentity, gid)
        assert ident.role_state == "unknown"
        assert next_global_id(s) == gid + 1
        # pgvector cosine search finds the track by its clip embedding
        hit = s.scalar(
            select(TrackEmbedding.track_id)
            .order_by(TrackEmbedding.clip.cosine_distance(np.eye(512)[0]))
            .limit(1)
        )
        assert hit == row.id
    db.dispose()
