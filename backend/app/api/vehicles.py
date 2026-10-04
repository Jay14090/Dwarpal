"""Vehicle registry, plate reads and recent events."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.anpr.normalize import normalize
from app.api.deps import actor, db_session
from app.db.models import AuditLog, Event, PlateRead, Vehicle

router = APIRouter(tags=["vehicles"])


class VehicleIn(BaseModel):
    plate: str = Field(min_length=4, max_length=20)
    vehicle_type: str | None = Field(None, max_length=32)
    owner_person_id: int | None = None


def _reload_registry(request: Request) -> None:
    proc = request.app.state.engine_proc
    if proc is not None and proc.alive:
        proc.request("reload_registry", timeout=15)


@router.get("/vehicles")
def list_vehicles(session: Annotated[Session, Depends(db_session)]) -> list[dict[str, Any]]:
    return [
        {
            "id": v.id,
            "plate": v.plate,
            "vehicle_type": v.vehicle_type,
            "owner_person_id": v.owner_person_id,
        }
        for v in session.scalars(select(Vehicle).order_by(Vehicle.plate))
    ]


@router.post("/vehicles")
def add_vehicle(
    body: VehicleIn,
    request: Request,
    session: Annotated[Session, Depends(db_session)],
    who: Annotated[str, Depends(actor)],
) -> dict[str, Any]:
    plate = normalize(body.plate)
    if not plate.valid:
        raise HTTPException(
            422,
            f"{body.plate!r} is not an Indian registration number (e.g. TN09AB1234, 22BH1234AB)",
        )
    v = Vehicle(
        plate=plate.text, vehicle_type=body.vehicle_type, owner_person_id=body.owner_person_id
    )
    session.add(v)
    try:
        session.flush()
    except IntegrityError as exc:
        raise HTTPException(409, f"{plate.text} is already registered") from exc
    session.add(
        AuditLog(
            actor=who,
            action="register_vehicle",
            target=f"vehicle:{v.id}",
            details={"plate": plate.text},
        )
    )
    session.commit()
    _reload_registry(request)
    return {
        "id": v.id,
        "plate": v.plate,
        "normalized_from": body.plate,
        "corrections": plate.corrections,
    }


@router.delete("/vehicles/{vehicle_id}")
def remove_vehicle(
    vehicle_id: int,
    request: Request,
    session: Annotated[Session, Depends(db_session)],
    who: Annotated[str, Depends(actor)],
) -> dict[str, Any]:
    v = session.get(Vehicle, vehicle_id)
    if v is None:
        raise HTTPException(404, "no such vehicle")
    session.delete(v)
    session.add(
        AuditLog(
            actor=who,
            action="remove_vehicle",
            target=f"vehicle:{vehicle_id}",
            details={"plate": v.plate},
        )
    )
    session.commit()
    _reload_registry(request)
    return {"deleted": vehicle_id}


@router.get("/plates")
def plate_reads(
    session: Annotated[Session, Depends(db_session)],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    plate: str | None = None,
) -> list[dict[str, Any]]:
    q = select(PlateRead).order_by(PlateRead.ts.desc()).limit(limit)
    if plate:
        q = q.where(PlateRead.plate_text == normalize(plate).text)
    return [
        {
            "id": r.id,
            "camera_id": r.camera_id,
            "ts": r.ts.isoformat(),
            "plate": r.plate_text,
            "raw": r.raw_text,
            "confidence": round(r.confidence, 3),
            "status": r.status,
            "vehicle_id": r.vehicle_id,
            "thumb_url": f"/plates/{r.id}/thumb.jpg" if r.thumb_path else None,
        }
        for r in session.scalars(q)
    ]


@router.get("/plates/{read_id}/thumb.jpg")
def plate_thumb(read_id: int, session: Annotated[Session, Depends(db_session)]) -> Response:
    r = session.get(PlateRead, read_id)
    if r is None or not r.thumb_path or not Path(r.thumb_path).is_file():
        raise HTTPException(404, "no thumbnail")
    return Response(Path(r.thumb_path).read_bytes(), media_type="image/jpeg")


@router.get("/events")
def events(
    session: Annotated[Session, Depends(db_session)],
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    rule: str | None = None,
) -> list[dict[str, Any]]:
    q = select(Event).order_by(Event.ts.desc()).limit(limit)
    if rule:
        q = q.where(Event.rule == rule)
    return [
        {
            "id": e.id,
            "rule": e.rule,
            "severity": e.severity,
            "camera_id": e.camera_id,
            "ts": e.ts.isoformat(),
            "global_id": e.payload.get("global_id"),
            "plate_read_id": e.plate_read_id,
            "payload": e.payload,
            "acknowledged": e.acknowledged,
        }
        for e in session.scalars(q)
    ]
