from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from app.db.models import AuditLog, Event, GlobalIdentity
from app.events import EventRecord
from app.main import create_app
from app.pipeline.engine import store_event

pytestmark = pytest.mark.db


@pytest.fixture
def api(migrated_db_url, config, tmp_path):
    object.__setattr__(config.settings.paths, "thumbs_dir", tmp_path / "thumbs")
    db = create_engine(migrated_db_url)
    with Session(db) as s:
        for m in (Event, AuditLog, GlobalIdentity):
            s.execute(delete(m))
        s.commit()
    app = create_app(config=config, engine=db, start_engine=False)
    with TestClient(app) as client:
        yield client, db, config
    db.dispose()


def test_store_event_links_identity_and_thumbnail(api):
    client, db, config = api
    cam = config.cameras.cameras[0].id
    ev = EventRecord("unknown_in_restricted_zone", "unknown_in_zone", "high", cam, time.time(), global_id=42,
                     payload={"zone": "lobby"}, thumb_jpeg=b"\xff\xd8thumb")  # fmt: skip
    with Session(db) as s:
        eid = store_event(s, ev, config.settings.paths.thumbs_dir)
        s.commit()
        assert s.get(GlobalIdentity, 42) is not None
    evs = client.get("/events").json()
    assert (
        evs[0]["id"] == eid
        and evs[0]["global_id"] == 42
        and evs[0]["rule_type"] == "unknown_in_zone"
    )
    assert evs[0]["thumb_url"] == f"/events/{eid}/thumb.jpg"
    assert client.get(evs[0]["thumb_url"]).content == b"\xff\xd8thumb"

    r = client.post(f"/events/{eid}/ack", headers={"X-Actor": "guard1"})
    assert r.status_code == 200 and r.json()["acknowledged"] is True
    client.post(f"/events/{eid}/ack", headers={"X-Actor": "guard1"})  # idempotent: one audit row
    assert client.get("/events", params={"unacknowledged": True}).json() == []
    assert client.post("/events/999999/ack").status_code == 404
    with Session(db) as s:
        audits = s.scalars(select(AuditLog).where(AuditLog.action == "ack_event")).all()
        assert [(a.actor, a.target) for a in audits] == [("guard1", f"event:{eid}")]


def test_websocket_pushes_hub_events(api):
    client, _, _ = api
    hub = client.app.state.frame_hub
    with client.websocket_connect("/ws/events") as ws:
        for _ in range(50):  # the listener registers after accept
            if hub.listeners:
                break
            time.sleep(0.02)
        hub.push("event", {"rule": "loitering_unknown", "global_id": 3})
        hub.push("plate", {"plate": "KA01MX5555", "status": "unregistered"})
        assert ws.receive_json() == {
            "kind": "event",
            "data": {"rule": "loitering_unknown", "global_id": 3},
        }
        assert ws.receive_json()["data"]["plate"] == "KA01MX5555"
    for _ in range(50):
        if not hub.listeners:
            break
        time.sleep(0.02)
    assert hub.listeners == []
