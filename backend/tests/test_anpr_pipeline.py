from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from app.anpr.hook import PlateHook, PlateReadCache, PlateReadRecorder
from app.anpr.ocr import OcrRead
from app.anpr.plate_detector import PlateBox
from app.anpr.registry import Registry
from app.anpr.service import PlateService
from app.events import EventBus
from app.pipeline.frames import Frame, FrameResult, Track

IMG = np.full((240, 320, 3), 128, np.uint8)


class FakePlateDetector:
    def detect(self, image):
        h, w = image.shape[:2]
        return [PlateBox((w * 0.3, h * 0.7, w * 0.7, h * 0.9), 0.9)]


class ScriptedOcr:
    """Returns the scripted texts in order (then repeats the last)."""

    def __init__(self, texts, conf=0.9):
        self.texts = iter(texts)
        self.last = None
        self.conf = conf

    def read(self, plates):
        out = []
        for _ in plates:
            self.last = next(self.texts, self.last)
            out.append(OcrRead(self.last, self.conf, self.conf))
        return out


def frame(i, tracks):
    return FrameResult(Frame("gate", i, 1000.0 + i / 30, IMG), tracks)


def car(tid=1):
    return Track(tid, (40.0, 40.0, 280.0, 200.0), 0.9, "car")


@pytest.fixture
def anpr_cfg(config):
    return config.settings.anpr.model_copy(update={"sample_every": 1})


def run(hook, n, tracks_fn=lambda i: [car()]):
    for i in range(n):
        hook(frame(i, tracks_fn(i)))


def test_noisy_reads_vote_to_the_right_plate(anpr_cfg):
    decisions = []
    ocr = ScriptedOcr(["TN09AB1234", "TN-O9AB1234", "TN09A81234", "TN09AB1234", "TN09AB1234"])
    hook = PlateHook(
        "gate",
        anpr_cfg,
        lambda d: decisions.append(d) or None,
        detector=FakePlateDetector(),
        ocr=ocr,
    )
    run(hook, 5)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.vote.plate == "TN09AB1234" and d.vote.reads >= 3
    assert d.thumb_jpeg[:2] == b"\xff\xd8" and d.vehicle_label == "car"


def test_low_confidence_and_invalid_reads_are_ignored(anpr_cfg):
    decisions = []
    hook = PlateHook("gate", anpr_cfg, lambda d: decisions.append(d) or None,
                     detector=FakePlateDetector(), ocr=ScriptedOcr(["HELLO"] * 10))  # fmt: skip
    run(hook, 10)
    hook2 = PlateHook("gate", anpr_cfg, lambda d: decisions.append(d) or None,
                      detector=FakePlateDetector(), ocr=ScriptedOcr(["TN09AB1234"] * 10, conf=0.1))  # fmt: skip
    run(hook2, 10)
    assert decisions == []


def test_undecided_track_settles_when_it_leaves(anpr_cfg):
    decisions = []
    # two reads only (below min_reads=3) then the car leaves
    hook = PlateHook("gate", anpr_cfg, lambda d: decisions.append(d) or None,
                     detector=FakePlateDetector(), ocr=ScriptedOcr(["KA05MJ8821"]), grace_frames=5)  # fmt: skip
    run(hook, 30, lambda i: [car()] if i < 2 else [])
    assert [d.vote.plate for d in decisions] == ["KA05MJ8821"]


def test_cached_reads_replay_identically(anpr_cfg, tmp_path):
    rec = PlateReadRecorder()
    live = []
    ocr = ScriptedOcr(["DL1CAB1234"] * 5)
    hook = PlateHook("gate", anpr_cfg, lambda d: live.append(d.vote.plate) or None,
                     detector=FakePlateDetector(), ocr=ocr, recorder=rec)  # fmt: skip
    run(hook, 5)
    rec.save(tmp_path / "plates.jsonl")
    replayed = []
    cached = PlateHook("gate", anpr_cfg, lambda d: replayed.append(d.vote.plate) or None,
                       cache=PlateReadCache(tmp_path / "plates.jsonl"))  # fmt: skip
    run(cached, 5)
    assert live == replayed == ["DL1CAB1234"]


def test_service_registry_statuses_and_unregistered_event_cooldown(config, tmp_path, anpr_cfg):
    events = []
    bus = EventBus(config.rules, events.append)
    service = PlateService(Registry({"TN09AB1234": 1}), tmp_path, bus)
    shown = []
    hook = PlateHook("gate", anpr_cfg, lambda d: shown.append(service.handle(d)) or shown[-1],
                     detector=FakePlateDetector(), ocr=ScriptedOcr(["TN09AB1234"] * 3 + ["MH12AB1234"] * 3 + ["TN09AB1284"] * 3))  # fmt: skip
    ids = itertools.chain([1] * 3, [2] * 3, [3] * 3)
    run(hook, 9, lambda i: [car(next(ids))])
    assert shown == ["TN09AB1234 reg", "MH12AB1234 UNREG", "TN09AB1284 verify"]
    assert [e.rule_type for e in events] == ["unregistered_vehicle"]
    assert events[0].payload["plate"] == "MH12AB1234"
    # same unregistered plate again within the cooldown: no second event
    hook2 = PlateHook(
        "gate",
        anpr_cfg,
        service.handle,
        detector=FakePlateDetector(),
        ocr=ScriptedOcr(["MH12AB1234"] * 3),
    )
    run(hook2, 3, lambda i: [car(9)])
    assert len(events) == 1
    assert {r["status"] for r in service.recent} == {
        "registered",
        "unregistered",
        "likely_registered",
    }
    assert (tmp_path / "plates").is_dir()


def test_track_label_shows_plate_and_status(config, tmp_path, anpr_cfg):
    service = PlateService(Registry(), tmp_path)
    hook = PlateHook(
        "gate",
        anpr_cfg,
        service.handle,
        detector=FakePlateDetector(),
        ocr=ScriptedOcr(["MH12AB1234"] * 4),
    )
    tracks = [car()]
    for i in range(4):
        tracks = [car()]
        hook(frame(i, tracks))
    assert tracks[0].extra["plate"] == "MH12AB1234 UNREG"


@pytest.mark.db
def test_plate_reads_and_events_are_stored(migrated_db_url, config, tmp_path, anpr_cfg):
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.db.models import Event, PlateRead
    from app.db.sync import sync_cameras
    from app.pipeline.engine import store_event
    from app.pipeline.indexer import DbWriter

    db = create_engine(migrated_db_url)
    writer = DbWriter(db)
    writer.submit(lambda s: sync_cameras(s, config.cameras))
    stored = []
    bus = EventBus(
        config.rules, lambda ev: writer.submit(lambda s: store_event(s, ev), stored.append)
    )
    service = PlateService(Registry(), tmp_path, bus, writer)
    cam = config.cameras.cameras[0].id
    hook = PlateHook(
        cam,
        anpr_cfg,
        service.handle,
        detector=FakePlateDetector(),
        ocr=ScriptedOcr(["UP32AB0001"] * 3),
    )
    run(hook, 3)
    deadline = time.monotonic() + 5
    while not stored and time.monotonic() < deadline:
        time.sleep(0.05)
    writer.flush()
    writer.close()
    with Session(db) as s:
        read = s.scalars(select(PlateRead).where(PlateRead.plate_text == "UP32AB0001")).one()
        assert read.status == "unregistered" and read.camera_id == cam
        ev = s.get(Event, stored[0])
        assert ev.plate_read_id == read.id and ev.payload["rule_type"] == "unregistered_vehicle"
    db.dispose()


def test_engine_anpr_camera_emits_plate_and_event(tmp_path, config):
    from app.core.config import Camera
    from app.pipeline.engine import Engine
    from app.pipeline.frames import Detections
    from app.pipeline.reid import ColorHistEncoder

    from .test_pipeline import engine_config, write_video

    class CarDetector:
        labels = ("person", "bicycle", "car", "motorcycle", "bus", "truck")

        def detect(self, images):
            return [Detections(np.array([[40, 40, 280, 200]], np.float32), np.array([0.9], np.float32),
                               np.array([2], np.int32), self.labels) for _ in images]  # fmt: skip

    video = write_video(tmp_path / "gate.mp4", n=30)
    cam = Camera(id="gate1", name="Gate", source_type="file", source_uri=str(video), anpr=True)
    cfg = engine_config(config, [cam])
    anpr = cfg.settings.anpr.model_copy(update={"sample_every": 1})
    cfg = cfg.model_copy(update={"settings": cfg.settings.model_copy(update={"anpr": anpr})})
    emitted: list[tuple[str, dict]] = []
    eng = Engine(cfg, lambda *a: None, detector=CarDetector(), encoder=ColorHistEncoder(),
                 plate_detector=FakePlateDetector(), plate_ocr=ScriptedOcr(["MH12AB1234"] * 100),
                 registry_loader=lambda: Registry({"TN09AB1234": 7}), emit=lambda k, p: emitted.append((k, p)))  # fmt: skip
    eng.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not any(k == "event" for k, _ in emitted):
        time.sleep(0.05)
    eng.stop()
    kinds = [k for k, _ in emitted]
    assert "plate" in kinds and "event" in kinds
    plate = next(p for k, p in emitted if k == "plate")
    assert plate["plate"] == "MH12AB1234" and plate["status"] == "unregistered"
    event = next(p for k, p in emitted if k == "event")
    assert event["rule_type"] == "unregistered_vehicle" and event["camera_id"] == "gate1"
    assert eng.stats()["anpr"]["registry"] == 1
