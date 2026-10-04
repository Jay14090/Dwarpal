"""Camera list, annotated MJPEG streams, snapshots and pipeline stats."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from app.pipeline.engine import FrameHub

router = APIRouter(tags=["cameras"])
BOUNDARY = "dwarpalframe"
ONLINE_AFTER_S = 3.0


def _hub(request: Request) -> FrameHub:
    return request.app.state.frame_hub


def _known(request: Request, camera_id: str) -> None:
    if camera_id not in {c.id for c in request.app.state.config.cameras.cameras}:
        raise HTTPException(404, f"unknown camera {camera_id}")


@router.get("/cameras")
def list_cameras(request: Request) -> list[dict[str, Any]]:
    hub = _hub(request)
    stats = {c["camera_id"]: c for c in hub.stats.get("cameras", [])}
    out = []
    for cam in request.app.state.config.cameras.cameras:
        latest = hub.get(cam.id)
        out.append(
            {
                "id": cam.id,
                "name": cam.name,
                "enabled": cam.enabled,
                "run_mode": cam.run_mode,
                "source_type": cam.source_type,
                "online": bool(latest and time.time() - latest[1] < ONLINE_AFTER_S),
                "zones": [z.model_dump() for z in cam.zones],
                "stats": stats.get(cam.id),
                "stream_url": f"/cameras/{cam.id}/stream.mjpg",
            }
        )
    return out


@router.get("/cameras/{camera_id}/snapshot.jpg")
def snapshot(camera_id: str, request: Request) -> Response:
    _known(request, camera_id)
    latest = _hub(request).get(camera_id)
    if latest is None:
        raise HTTPException(503, "no frame yet")
    return Response(latest[2], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/cameras/{camera_id}/frame.json")
def frame_meta(camera_id: str, request: Request) -> dict[str, Any]:
    """Tracks of the latest published frame (normalized boxes)."""
    _known(request, camera_id)
    latest = _hub(request).get(camera_id)
    if latest is None:
        raise HTTPException(503, "no frame yet")
    return latest[3]


async def _mjpeg(
    hub: FrameHub, camera_id: str, request: Request, poll_s: float
) -> AsyncIterator[bytes]:
    last = -1
    while not await request.is_disconnected():
        latest = hub.get(camera_id)
        if latest is not None and latest[0] != last:
            last = latest[0]
            jpeg = latest[2]
            yield (
                f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                + jpeg
                + b"\r\n"
            )
        await asyncio.sleep(poll_s)


@router.get("/cameras/{camera_id}/stream.mjpg")
def stream(camera_id: str, request: Request) -> StreamingResponse:
    _known(request, camera_id)
    fps = request.app.state.config.settings.streaming.mjpeg_max_fps
    return StreamingResponse(
        _mjpeg(_hub(request), camera_id, request, poll_s=1 / (2 * fps)),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/pipeline/stats")
def pipeline_stats(request: Request) -> dict[str, Any]:
    engine = request.app.state.engine_proc
    return {"engine_running": bool(engine and engine.alive), **_hub(request).stats}


LIVE_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Dwarpal live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{background:#0b0f14;color:#cfd8e3;font:14px system-ui;margin:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}
figure{margin:0;background:#121821;border:1px solid #1f2a36;border-radius:8px;overflow:hidden}
img{width:100%;display:block;background:#000;aspect-ratio:16/9;object-fit:contain}
figcaption{padding:6px 10px;display:flex;justify-content:space-between}
.s{color:#7f8c9a;font-variant-numeric:tabular-nums}</style></head><body>
<h3>Dwarpal live (P2 debug view; the full dashboard is the Next.js app)</h3><div class="grid" id="g"></div>
<script>
async function load(){const cams=await (await fetch('/cameras')).json();
const g=document.getElementById('g');
for(const c of cams.filter(c=>c.enabled)){const f=document.createElement('figure');
f.innerHTML=`<img src="${c.stream_url}"><figcaption><b>${c.name}</b><span class="s" id="s-${c.id}"></span></figcaption>`;g.appendChild(f);}
setInterval(async()=>{for(const c of await (await fetch('/cameras')).json()){const e=document.getElementById('s-'+c.id);
if(e){const s=c.stats;e.textContent=(c.online?'online':'offline')+(s?` · ${s.mode} · ${s.process_fps} fps`+(s.infer_ms?` · ${s.infer_ms} ms`:'')+` · ${s.tracks} tracks`:'');}}},2000);}
load();</script></body></html>"""


@router.get("/live", response_class=HTMLResponse, include_in_schema=False)
def live_page() -> str:
    return LIVE_PAGE
