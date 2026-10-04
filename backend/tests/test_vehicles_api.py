from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import AuditLog, Event, PlateRead, Vehicle
from app.main import create_app

pytestmark = pytest.mark.db


@pytest.fixture
def api(migrated_db_url, config):
    db = create_engine(migrated_db_url)
    with Session(db) as s:
        for m in (Event, PlateRead, Vehicle, AuditLog):
            s.query(m).delete()
        s.commit()
    app = create_app(config=config, engine=db, start_engine=False)
    with TestClient(app) as client:  # runs lifespan: cameras synced
        yield client, db
    db.dispose()


def test_register_vehicle_normalizes_and_validates(api):
    client, _ = api
    r = client.post("/vehicles", json={"plate": "tn-o9 ab 1234", "vehicle_type": "car"})
    assert r.status_code == 200 and r.json()["plate"] == "TN09AB1234"
    assert client.post("/vehicles", json={"plate": "TN09AB1234"}).status_code == 409
    assert client.post("/vehicles", json={"plate": "HELLO WORLD"}).status_code == 422
    assert [v["plate"] for v in client.get("/vehicles").json()] == ["TN09AB1234"]
    vid = r.json()["id"]
    assert client.delete(f"/vehicles/{vid}").status_code == 200
    assert client.get("/vehicles").json() == []


def test_plate_reads_and_events_listing(api, config):
    client, db = api
    cam = config.cameras.cameras[0].id  # synced into the cameras table by the lifespan hook
    with Session(db) as s:
        pr = PlateRead(
            camera_id=cam,
            ts=datetime.now(UTC),
            plate_text="MH12AB1234",
            confidence=0.8,
            status="unregistered",
        )
        s.add(pr)
        s.flush()
        s.add(Event(rule="unregistered_vehicle_at_gate", severity="medium", camera_id=cam, ts=datetime.now(UTC),
                    plate_read_id=pr.id, payload={"plate": "MH12AB1234"}))  # fmt: skip
        s.commit()
    reads = client.get("/plates", params={"plate": "mh 12 ab 1234"}).json()
    assert reads[0]["status"] == "unregistered" and reads[0]["thumb_url"] is None
    evs = client.get("/events").json()
    assert (
        evs[0]["rule"] == "unregistered_vehicle_at_gate"
        and evs[0]["plate_read_id"] == reads[0]["id"]
    )
