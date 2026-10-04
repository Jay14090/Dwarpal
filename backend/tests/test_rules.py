from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from app.core.config import Camera, RulesConfig
from app.events import EventBus, EventRecord
from app.pipeline.frames import Frame, FrameResult, Track
from app.rules import RulesEngine, in_window

IST = ZoneInfo("Asia/Kolkata")
LEFT = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
RIGHT = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]
IMG = np.zeros((100, 200, 3), np.uint8)
IN_LEFT = (40.0, 20.0, 60.0, 90.0)  # foot point x=0.25
IN_RIGHT = (140.0, 20.0, 160.0, 90.0)  # foot point x=0.75


def camera() -> Camera:
    return Camera(
        id="cam",
        name="Cam",
        source_type="file",
        source_uri="x.mp4",
        zones=[
            {"name": "lobby", "restricted": True, "polygon": LEFT},
            {"name": "garden", "restricted": False, "polygon": RIGHT},
        ],
    )


def rules(*specs: dict, cooldown: int = 300) -> RulesConfig:
    return RulesConfig.model_validate(
        {"defaults": {"cooldown_s": cooldown, "presence_gap_s": 5}, "rules": list(specs)}
    )


def setup(*specs: dict, cooldown: int = 300) -> tuple[RulesEngine, list[EventRecord]]:
    out: list[EventRecord] = []
    cfg = rules(*specs, cooldown=cooldown)
    return RulesEngine(
        cfg, EventBus(cfg, out.append), "Asia/Kolkata", cfg.defaults.presence_gap_s
    ), out


def at(hh: int, mm: int = 0, ss: float = 0.0) -> float:
    return datetime(2026, 10, 4, hh, mm, tzinfo=IST).timestamp() + ss


def step(
    engine: RulesEngine, ts: float, *people: tuple[int | None, str, tuple], track_base: int = 0
) -> None:
    tracks = [
        Track(track_id=track_base + i, xyxy=box, conf=0.9, label="person", global_id=gid, role=role)
        for i, (gid, role, box) in enumerate(people)
    ]
    engine.observe(camera(), FrameResult(Frame("cam", 0, ts, IMG), tracks))


UNKNOWN = {"id": "u", "type": "unknown_in_zone", "severity": "high", "zones": []}
LOITER = {"id": "l", "type": "loitering", "zones": [], "params": {"min_dwell_s": 60}}


def test_in_window_crosses_midnight():
    from datetime import time

    assert in_window(time(23, 0), time(22, 0), time(6, 0))
    assert in_window(time(2, 0), time(22, 0), time(6, 0))
    assert not in_window(time(12, 0), time(22, 0), time(6, 0))
    assert in_window(time(10, 0), time(9, 0), time(17, 0)) and not in_window(
        time(17, 0), time(9, 0), time(17, 0)
    )


def test_unknown_in_zone_fires_once_per_incident():
    engine, out = setup(UNKNOWN)
    t0 = at(12)
    for k in range(200):  # 200 s in the zone at 1 fps
        step(engine, t0 + k, (7, "unknown", IN_LEFT))
    assert len(out) == 1
    ev = out[0]
    assert (ev.rule, ev.global_id, ev.camera_id, ev.payload["zone"]) == ("u", 7, "cam", "lobby")
    assert ev.thumb_jpeg and ev.thumb_jpeg[:2] == b"\xff\xd8" and "thumb_jpeg" not in ev.as_dict()


def test_unknown_rule_ignores_other_roles_and_unrestricted_zones():
    engine, out = setup(UNKNOWN)
    for k in range(10):
        step(engine, at(12) + k, (1, "resident", IN_LEFT), (2, "pending", IN_LEFT), (3, "staff", IN_LEFT),
             (4, "unknown", IN_RIGHT))  # fmt: skip
    assert out == []


def test_pending_then_unknown_fires_when_resolved():
    engine, out = setup(UNKNOWN)
    for k in range(5):
        step(engine, at(12) + k, (7, "pending", IN_LEFT))
    assert out == []
    step(engine, at(12) + 5, (7, "unknown", IN_LEFT))
    assert len(out) == 1


def test_short_detection_gaps_do_not_split_an_incident():
    engine, out = setup(UNKNOWN, cooldown=0)
    t0 = at(12)
    for k in range(30):
        if k % 4 != 3:  # missed every 4th second (gap 2 s < presence_gap_s 5)
            step(engine, t0 + k, (7, "unknown", IN_LEFT))
    assert len(out) == 1


def test_new_incident_after_leaving_respects_cooldown():
    engine, out = setup(UNKNOWN, cooldown=300)
    t0 = at(12)
    step(engine, t0, (7, "unknown", IN_LEFT))
    step(
        engine, t0 + 30, (7, "unknown", IN_LEFT)
    )  # left > 5 s and came back: same person, in cooldown
    assert len(out) == 1
    step(engine, t0 + 400, (7, "unknown", IN_LEFT))  # new incident after the cooldown
    assert len(out) == 2
    step(engine, t0 + 401, (8, "unknown", IN_LEFT))  # someone else: own dedup key
    assert len(out) == 3


def test_loitering_needs_dwell_and_fires_once():
    engine, out = setup(LOITER)
    t0 = at(12)
    for k in range(59):
        step(engine, t0 + k, (7, "unknown", IN_LEFT))
    assert out == []
    for k in range(59, 300):
        step(engine, t0 + k, (7, "unknown", IN_LEFT))
    assert len(out) == 1 and out[0].payload["dwell_s"] == 60.0
    # walking out and back resets the dwell clock
    engine2, out2 = setup(LOITER)
    for k in range(0, 100, 1):
        if 40 <= k < 50:
            continue  # out of view for 10 s > gap
        step(engine2, t0 + k, (7, "unknown", IN_LEFT))
    assert out2 == []


def test_after_hours_spares_staff_and_daytime():
    spec = {
        "id": "ah",
        "type": "after_hours",
        "zones": ["lobby"],
        "params": {"start": "22:00", "end": "06:00"},
    }
    engine, out = setup(spec)
    step(engine, at(14), (1, "resident", IN_LEFT))
    step(engine, at(23), (2, "staff", IN_LEFT))
    assert out == []
    step(engine, at(23, 30), (3, "resident", IN_LEFT), (4, "unknown", IN_LEFT))
    assert sorted(e.global_id for e in out) == [3, 4]


def test_tailgating_unknown_right_after_resident():
    spec = {"id": "tg", "type": "tailgating", "zones": ["lobby"], "params": {"window_s": 3}}
    engine, out = setup(spec)
    t0 = at(12)
    step(engine, t0, (1, "resident", IN_LEFT))
    step(engine, t0 + 2, (1, "resident", IN_LEFT), (9, "unknown", IN_LEFT))
    assert [e.global_id for e in out] == [9]
    engine, out = setup(spec)
    step(engine, t0, (1, "resident", IN_LEFT))
    step(engine, t0 + 4.5, (1, "resident", IN_LEFT), (9, "unknown", IN_LEFT))  # too late
    assert out == []


def test_tracks_without_global_id_use_camera_track_key():
    engine, out = setup(UNKNOWN)
    step(engine, at(12), (None, "unknown", IN_LEFT), track_base=5)
    step(engine, at(12) + 1, (None, "unknown", IN_LEFT), track_base=5)
    assert len(out) == 1 and out[0].global_id is None


def test_disabled_rules_and_vehicle_rules_are_skipped():
    engine, out = setup({**UNKNOWN, "enabled": False}, {"id": "v", "type": "unregistered_vehicle"})
    step(engine, at(12), (7, "unknown", IN_LEFT))
    assert out == [] and engine.rules == []
