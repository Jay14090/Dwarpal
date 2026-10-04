from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.api.cameras import BOUNDARY, _mjpeg
from app.main import create_app
from app.pipeline.engine import FrameHub

JPEG = b"\xff\xd8fakejpeg\xff\xd9"


def client(config) -> TestClient:
    app = create_app(config=config, start_engine=False)
    app.state.frame_hub.put("webcam", JPEG, {"camera_id": "webcam", "tracks": []})
    app.state.frame_hub.stats = {
        "cameras": [{"camera_id": "webcam", "mode": "realtime", "process_fps": 12.0}]
    }
    return TestClient(app)


def test_list_cameras(config):
    cams = {c["id"]: c for c in client(config).get("/cameras").json()}
    assert set(cams) == {c.id for c in config.cameras.cameras}
    assert cams["webcam"]["online"] is True
    assert cams["webcam"]["stats"]["process_fps"] == 12.0
    assert cams["meva_g336"]["online"] is False


def test_snapshot_and_meta(config):
    c = client(config)
    r = c.get("/cameras/webcam/snapshot.jpg")
    assert r.status_code == 200 and r.content == JPEG and r.headers["content-type"] == "image/jpeg"
    assert c.get("/cameras/webcam/frame.json").json()["camera_id"] == "webcam"
    assert c.get("/cameras/meva_g336/snapshot.jpg").status_code == 503
    assert c.get("/cameras/nope/snapshot.jpg").status_code == 404


def test_live_page_and_stats(config):
    c = client(config)
    assert "Dwarpal live" in c.get("/live").text
    assert c.get("/pipeline/stats").json()["engine_running"] is False


class FakeRequest:
    def __init__(self, polls: int) -> None:
        self.polls = polls

    async def is_disconnected(self) -> bool:
        self.polls -= 1
        return self.polls < 0


def test_mjpeg_generator_yields_each_new_frame_once():
    hub = FrameHub()
    hub.put("cam", JPEG, {})

    async def collect():
        return [chunk async for chunk in _mjpeg(hub, "cam", FakeRequest(polls=3), poll_s=0)]

    chunks = asyncio.run(collect())
    assert len(chunks) == 1  # same frame version is not re-sent
    assert chunks[0].startswith(f"--{BOUNDARY}\r\nContent-Type: image/jpeg".encode())
    assert JPEG in chunks[0]
