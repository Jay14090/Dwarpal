"""Events (P8): history, acknowledgement, thumbnails and the live WebSocket feed."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import actor, db_session
from app.api.privacy import serve_thumb
from app.db.models import AuditLog, Event, GlobalIdentity
from app.pipeline.engine import event_thumb_path

log = logging.getLogger(__name__)
router = APIRouter(tags=["events"])


def event_dict(e: Event) -> dict[str, Any]:
    return {
        "id": e.id,
        "rule": e.rule,
        "rule_type": e.payload.get("rule_type"),
        "severity": e.severity,
        "camera_id": e.camera_id,
        "ts": e.ts.isoformat(),
        "global_id": e.global_id if e.global_id is not None else e.payload.get("global_id"),
        "plate_read_id": e.plate_read_id,
        "payload": e.payload,
        "acknowledged": e.acknowledged,
        "thumb_url": f"/events/{e.id}/thumb.jpg" if e.payload.get("thumb") else None,
    }


@router.get("/events")
def events(
    session: Annotated[Session, Depends(db_session)],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    rule: str | None = None,
    unacknowledged: bool = False,
) -> list[dict[str, Any]]:
    q = select(Event).order_by(Event.ts.desc(), Event.id.desc()).limit(limit)
    if rule:
        q = q.where(Event.rule == rule)
    if unacknowledged:
        q = q.where(Event.acknowledged.is_(False))
    return [event_dict(e) for e in session.scalars(q)]


@router.post("/events/{event_id}/ack")
def acknowledge(
    event_id: int,
    session: Annotated[Session, Depends(db_session)],
    who: Annotated[str, Depends(actor)],
) -> dict[str, Any]:
    e = session.get(Event, event_id)
    if e is None:
        raise HTTPException(404, "event not found")
    if not e.acknowledged:
        e.acknowledged = True
        session.add(
            AuditLog(
                actor=who, action="ack_event", target=f"event:{e.id}", details={"rule": e.rule}
            )
        )
    session.flush()
    return event_dict(e)


@router.get("/events/{event_id}/thumb.jpg")
def event_thumb(
    event_id: int,
    request: Request,
    session: Annotated[Session, Depends(db_session)],
    who: Annotated[str, Depends(actor)],
    unblur: bool = False,
) -> Response:
    """Person crop of the alert; head blurred unless the person is identified (admin unblur is audited)."""
    path = event_thumb_path(request.app.state.config.settings.paths.thumbs_dir, event_id)
    e = session.get(Event, event_id)
    if e is None or not path.is_file():
        raise HTTPException(404, "no thumbnail")
    gi = session.get(GlobalIdentity, e.global_id) if e.global_id is not None else None
    role = (gi.role_state if gi else None) or e.payload.get("role") or "pending"
    data = serve_thumb(path.read_bytes(), role, request, session, unblur, who, f"event:{event_id}")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.websocket("/ws/events")
async def events_ws(ws: WebSocket) -> None:
    """Pushes {"kind": "event"|"plate", "data": {...}} as the engine produces them."""
    await ws.accept()
    hub = ws.app.state.frame_hub
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(maxsize=1000)

    def listener(kind: str, payload: dict[str, Any]) -> None:  # engine drain thread
        def put() -> None:
            with contextlib.suppress(asyncio.QueueFull):  # a stalled client loses old messages
                q.put_nowait((kind, payload))

        loop.call_soon_threadsafe(put)

    async def pump() -> None:
        while True:
            kind, payload = await q.get()
            await ws.send_json({"kind": kind, "data": payload})

    hub.listeners.append(listener)
    sender = asyncio.create_task(pump())
    try:
        while True:  # client messages are ignored; this notices the disconnect
            await ws.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.listeners.remove(listener)
        sender.cancel()
