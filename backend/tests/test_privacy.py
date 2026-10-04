from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from app.clips import blur_fn, cut_clip
from app.core.config import Camera, PrivacySettings
from app.db.models import AuditLog, Event, GlobalIdentity, Track, TrackEmbedding
from app.main import create_app
from app.pipeline.annotate import Annotator
from app.pipeline.cache import TrackCache, TrackCacheWriter, cache_path
from app.pipeline.frames import Frame, FrameResult
from app.pipeline.frames import Track as LiveTrack
from app.pipeline.worker import CameraWorker
from app.privacy import PrivacyPolicy, blur_crop, blur_heads, head_box, run_retention

RNG = np.random.default_rng(0)
BOX = (100.0, 60.0, 160.0, 220.0)  # person box: head ~ y 60..98


def texture(h: int = 240, w: int = 320) -> np.ndarray:
    return RNG.integers(0, 255, (h, w, 3), dtype=np.uint8)


def head_std(img: np.ndarray, box=BOX) -> float:
    x1, y1, x2, y2 = head_box(box, img.shape)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return float(img[cy - 6 : cy + 6, cx - 10 : cx + 10].std())


def body_std(img: np.ndarray, box=BOX) -> float:
    x1, y1, x2, y2 = (int(v) for v in box)
    cy = int(y1 + 0.7 * (y2 - y1))
    cx = (x1 + x2) // 2
    return float(img[cy - 6 : cy + 6, cx - 10 : cx + 10].std())


# ---------------------------------------------------------------- blur primitives


def test_blur_heads_only_touches_the_head():
    img = texture()
    out = blur_heads(img, [BOX])
    assert head_std(img) > 50 and head_std(out) < 15  # head unreadable
    assert np.array_equal(out[150:], img[150:])  # body and rest of frame untouched
    assert not np.array_equal(out, img) and np.array_equal(img, blur_heads(img, []))


def test_blur_crop_and_head_box_bounds():
    crop = texture(192, 80)
    out = blur_crop(crop)
    assert out[:20].std() < 15 and np.array_equal(out[100:], crop[100:])
    assert head_box((0, 0, 10, 10), (5, 5, 3)) == (0, 0, 5, 3)  # clipped to the image


def test_policy_roles():
    pol = PrivacyPolicy(PrivacySettings())
    assert pol.should_blur("unknown") and pol.should_blur("pending") and pol.should_blur(None)
    assert not pol.should_blur("resident") and not pol.should_blur("staff")
    assert not PrivacyPolicy(PrivacySettings(blur_unknown_faces=False)).should_blur("unknown")
    assert pol.is_admin("admin") and not pol.is_admin("operator")


# ---------------------------------------------------------------- live stream


class _Cache:  # any non-None cache satisfies the worker; tracks are passed in directly
    pass


def published_frame(config, role: str, unblur_s: float = 0.0) -> np.ndarray:
    cam = Camera(id="c", name="C", source_type="file", source_uri="x.mp4")
    out = []
    w = CameraWorker(cam, None, lambda c, j, m: out.append(j), Annotator(config.settings.pipeline.annotate),  # type: ignore[arg-type]
                     cache=_Cache(), publish_fps=1e9, jpeg_quality=100,
                     privacy=PrivacyPolicy(config.settings.privacy))  # type: ignore[arg-type]  # fmt: skip
    w.unblur_until = time.monotonic() + unblur_s
    t = LiveTrack(track_id=1, xyxy=BOX, conf=0.9, label="person", global_id=1, role=role)
    w._publish(FrameResult(Frame("c", 0, time.time(), texture()), [t]))
    return cv2.imdecode(np.frombuffer(out[-1], np.uint8), cv2.IMREAD_COLOR)


def test_stream_blurs_unknown_and_pending_but_not_residents(config):
    assert head_std(published_frame(config, "unknown")) < 20
    assert head_std(published_frame(config, "pending")) < 20
    assert head_std(published_frame(config, "resident")) > 40
    assert body_std(published_frame(config, "unknown")) > 40  # only the head


def test_stream_unblur_window(config):
    assert head_std(published_frame(config, "unknown", unblur_s=60)) > 40
    assert head_std(published_frame(config, "unknown", unblur_s=-1)) < 20  # expired


# ---------------------------------------------------------------- clips


def write_video(path, n=30, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    frames = [texture(size[1], size[0]) for _ in range(n)]
    for f in frames:
        w.write(f)
    w.release()
    return path


def make_cache(path, n=30, track_id=1):
    wr = TrackCacheWriter(("person",))
    for f in range(n):
        wr.add(f, [LiveTrack(track_id=track_id, xyxy=BOX, conf=0.9, label="person")])
    wr.save(path, {"num_frames": n, "fps": 30.0})
    return TrackCache(path)


def middle_frame(path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ok, img = cap.read()
    cap.release()
    assert ok
    return img


def test_clip_blurs_unknown_and_keeps_identified_person(tmp_path):
    video = write_video(tmp_path / "v.mp4")
    cache = make_cache(tmp_path / "tracks.npz")
    blurred = cut_clip(
        video, tmp_path / "b.mp4", 5, 25, blur_fn(cache, keep_track_id=None), pad_s=0
    )
    kept = cut_clip(video, tmp_path / "k.mp4", 5, 25, blur_fn(cache, keep_track_id=1), pad_s=0)
    assert head_std(middle_frame(blurred)) < 20
    assert head_std(middle_frame(kept)) > 40


# ---------------------------------------------------------------- API + retention (Postgres)


JPEG = cv2.imencode(".jpg", texture(192, 80))[1].tobytes()
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


@pytest.fixture
def api(migrated_db_url, config, tmp_path):
    s = config.settings
    for k in ("thumbs_dir", "clips_dir", "cache_dir"):
        object.__setattr__(s.paths, k, tmp_path / k)
    db = create_engine(migrated_db_url)
    with Session(db) as ses:
        for m in (TrackEmbedding, Event, Track, GlobalIdentity, AuditLog):
            ses.execute(delete(m))
        ses.commit()
    app = create_app(config=config, engine=db, start_engine=False)
    with TestClient(app) as client:
        yield client, db, config
    db.dispose()


def add_track(ses, config, tid, gid, role, end, thumb=True):
    thumbs = config.settings.paths.thumbs_dir / "tracks"
    thumbs.mkdir(parents=True, exist_ok=True)
    p = thumbs / f"{tid}.jpg"
    if thumb:
        p.write_bytes(JPEG)
    if gid is not None and ses.get(GlobalIdentity, gid) is None:
        ses.add(GlobalIdentity(id=gid, role_state=role, first_seen=end, last_seen=end))
        ses.flush()
    ses.add(Track(id=tid, global_id=gid, camera_id="meva_g336", local_track_id=1, start_ts=end - timedelta(seconds=10),
                  end_ts=end, role=role, thumb_path=str(p) if thumb else None, start_frame=5, end_frame=25))  # fmt: skip
    ses.flush()
    ses.add(TrackEmbedding(track_id=tid, clip=[0.1] * 512, body=[0.1] * 512))


def test_thumbnails_blurred_by_default_admin_unblur_audited(api):
    client, db, config = api
    with Session(db) as ses:
        add_track(ses, config, 1, 11, "unknown", NOW)
        add_track(ses, config, 2, 12, "resident", NOW)
        ses.commit()
    unknown = cv2.imdecode(
        np.frombuffer(client.get("/tracks/1/thumb.jpg").content, np.uint8), cv2.IMREAD_COLOR
    )
    assert unknown[:20].std() < 20
    assert client.get("/tracks/2/thumb.jpg").content == JPEG  # identified: as stored
    assert client.get("/tracks/1/thumb.jpg", params={"unblur": True}).status_code == 403  # operator
    r = client.get("/tracks/1/thumb.jpg", params={"unblur": True}, headers={"X-Actor": "admin"})
    assert r.content == JPEG
    # becoming resident later (enrollment) unblurs the history without admin action
    with Session(db) as ses:
        ses.get(GlobalIdentity, 11).role_state = "resident"
        ses.commit()
    assert client.get("/tracks/1/thumb.jpg").content == JPEG
    audits = client.get("/audit", headers={"X-Actor": "admin"}).json()
    assert [(a["actor"], a["action"], a["target"]) for a in audits] == [
        ("admin", "unblur_thumbnail", "track:1")
    ]
    assert client.get("/audit").status_code == 403


def test_clip_endpoint_blurs_and_audits_unblur(api, tmp_path):
    client, db, config = api
    s = config.settings
    video = write_video(tmp_path / "cam.mp4")
    cam = next(c for c in config.cameras.cameras if c.id == "meva_g336")
    object.__setattr__(cam, "source_uri", str(video))
    with Session(db) as ses:
        add_track(ses, config, 1, 11, "unknown", NOW)
        ses.commit()
    assert client.get("/tracks/1/clip.mp4").status_code == 409  # no detections: cannot anonymise
    cp = cache_path(s.paths.cache_dir, "meva_g336")
    cp.parent.mkdir(parents=True)
    make_cache(cp)
    r = client.get("/tracks/1/clip.mp4")
    assert r.status_code == 200
    (tmp_path / "got.mp4").write_bytes(r.content)
    assert head_std(middle_frame(tmp_path / "got.mp4")) < 20
    r = client.get("/tracks/1/clip.mp4", params={"unblur": True}, headers={"X-Actor": "admin"})
    (tmp_path / "raw.mp4").write_bytes(r.content)
    assert head_std(middle_frame(tmp_path / "raw.mp4")) > 40
    assert not list((s.paths.clips_dir / "tmp").glob("*.mp4"))  # unblurred clips are not kept
    with Session(db) as ses:
        assert ses.scalars(select(AuditLog.action)).all() == ["unblur_clip"]


def test_retention_deletes_old_unknown_data_only(api):
    client, db, config = api
    s = config.settings
    old, recent = NOW - timedelta(days=8), NOW - timedelta(days=2)
    with Session(db) as ses:
        add_track(ses, config, 1, 11, "unknown", old)
        add_track(ses, config, 2, None, "pending", old)
        add_track(ses, config, 3, 13, "unknown", recent)
        add_track(ses, config, 4, 14, "resident", old)
        ev = Event(rule="unknown_in_restricted_zone", severity="high", camera_id="meva_g336", ts=old, global_id=11,
                   payload={"thumb": True})  # fmt: skip
        ses.add(ev)
        ses.commit()
        ev_id = ev.id
    ev_thumb = s.paths.thumbs_dir / "events" / f"{ev_id}.jpg"
    ev_thumb.parent.mkdir(parents=True)
    ev_thumb.write_bytes(JPEG)
    clip = s.paths.clips_dir / "tracks" / "1_meva_g336_5_25_b.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"x")
    with Session(db) as ses:
        rep = run_retention(ses, s.privacy, s.paths.thumbs_dir, s.paths.clips_dir, now=NOW)
    assert (rep.tracks, rep.embeddings, rep.events) == (2, 2, 1) and rep.files == 4
    with Session(db) as ses:
        assert sorted(ses.scalars(select(TrackEmbedding.track_id)).all()) == [3, 4]
        assert {t.id: t.thumb_path is None for t in ses.scalars(select(Track))} == {
            1: True,
            2: True,
            3: False,
            4: False,
        }
        assert ses.get(Event, ev_id).payload["thumb"] is False
        a = ses.scalars(select(AuditLog).where(AuditLog.action == "retention")).one()
        assert a.actor == "retention-job" and a.details["tracks"] == 2
    assert not clip.exists() and not ev_thumb.exists()
    assert (s.paths.thumbs_dir / "tracks" / "3.jpg").exists() and (
        s.paths.thumbs_dir / "tracks" / "4.jpg"
    ).exists()
    # the API trigger is admin-only and audited under the admin's name
    assert client.post("/privacy/retention").status_code == 403
    assert client.post("/privacy/retention", headers={"X-Actor": "admin"}).status_code == 200


def test_stream_unblur_endpoint_requires_admin_and_engine(api):
    client, _, _ = api
    body = {"camera_id": "webcam", "seconds": 30, "reason": "police request #42"}
    assert client.post("/privacy/unblur-stream", json=body).status_code == 403
    assert (
        client.post("/privacy/unblur-stream", json=body, headers={"X-Actor": "admin"}).status_code
        == 503
    )
