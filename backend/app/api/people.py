"""Enrollment (webcam capture or photo upload) and the people registry."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.deps import actor, db_session, engine_proc
from app.db.gallery_store import ConsentRequired, delete_person, enroll_person, list_people
from app.pipeline.engine import EngineProcess

router = APIRouter(tags=["people"])


class EnrollCapture(BaseModel):
    camera_id: str = "webcam"
    role: Literal["resident", "staff"]
    display_name: str = Field(min_length=1, max_length=128)
    unit: str | None = Field(None, max_length=64)
    consent: bool = False
    seconds: float | None = Field(None, gt=0, le=30)
    shots: int | None = Field(None, ge=1, le=20)


def _thumb_path(request: Request, person_id: int) -> Path:
    return request.app.state.config.settings.paths.thumbs_dir / "people" / f"{person_id}.jpg"


def _store(
    request: Request,
    session: Session,
    proc: EngineProcess,
    who: str,
    *,
    role: str,
    display_name: str,
    unit: str | None,
    consent: bool,
    shots: dict[str, Any],
    thumb: bytes | None,
) -> dict[str, Any]:
    min_shots = request.app.state.config.settings.identity.enrollment.min_shots
    n_face, n_body = len(shots.get("face", [])), len(shots.get("body", []))
    if n_face < min_shots and n_body < min_shots:
        raise HTTPException(
            422,
            f"only {n_face} face / {n_body} body shots of good quality (need {min_shots}); "
            "stand closer to the camera, face it, and try again",
        )
    try:
        person = enroll_person(
            session, role=role, display_name=display_name, unit=unit, consent=consent,
            face=shots.get("face", []) if n_face >= min_shots else [],
            body=shots.get("body", []),
            face_quality=shots.get("face_quality") if n_face >= min_shots else None,
            body_quality=shots.get("body_quality"), actor=who,
        )  # fmt: skip
    except ConsentRequired as exc:
        raise HTTPException(422, str(exc)) from exc
    session.commit()
    if thumb:
        path = _thumb_path(request, person.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(thumb)
    reload = proc.request("reload_gallery", timeout=15)
    return {
        "id": person.id,
        "role": person.role,
        "display_name": person.display_name,
        "unit": person.unit,
        "face_shots": n_face if n_face >= min_shots else 0,
        "body_shots": n_body,
        "gallery_people": reload.get("people"),
    }


@router.post("/enroll/capture")
def enroll_capture(
    body: EnrollCapture,
    request: Request,
    session: Annotated[Session, Depends(db_session)],
    proc: Annotated[EngineProcess, Depends(engine_proc)],
    who: Annotated[str, Depends(actor)],
) -> dict[str, Any]:
    """Capture the most prominent person on `camera_id` for a few seconds and enroll them."""
    if not body.consent:
        raise HTTPException(422, "consent is required to enroll a person")
    cfg = request.app.state.config.settings.identity.enrollment
    seconds = body.seconds or cfg.capture_seconds
    shots = proc.request(
        "enroll_capture", timeout=seconds + 20, camera_id=body.camera_id,
        seconds=seconds, shots=body.shots or cfg.capture_shots,
    )  # fmt: skip
    if "error" in shots:
        raise HTTPException(409, shots["error"])
    return _store(
        request, session, proc, who, role=body.role, display_name=body.display_name,
        unit=body.unit, consent=body.consent, shots=shots, thumb=shots.get("thumb_jpeg"),
    )  # fmt: skip


@router.post("/enroll/upload")
async def enroll_upload(
    request: Request,
    session: Annotated[Session, Depends(db_session)],
    proc: Annotated[EngineProcess, Depends(engine_proc)],
    who: Annotated[str, Depends(actor)],
    role: Annotated[Literal["resident", "staff"], Form()],
    display_name: Annotated[str, Form(min_length=1, max_length=128)],
    images: Annotated[list[UploadFile], File()],
    consent: Annotated[bool, Form()] = False,
    unit: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Enroll from 3-5 uploaded photos (face photos, optionally full-body photos)."""
    if not consent:
        raise HTTPException(422, "consent is required to enroll a person")
    blobs = [await f.read() for f in images[:10]]
    shots = await run_in_threadpool(proc.request, "embed_images", 60, images=blobs)
    if "error" in shots:
        raise HTTPException(400, shots["error"])
    return await run_in_threadpool(
        lambda: _store(
            request,
            session,
            proc,
            who,
            role=role,
            display_name=display_name,
            unit=unit,
            consent=consent,
            shots=shots,
            thumb=blobs[0] if blobs else None,
        )
    )


@router.get("/people")
def people(session: Annotated[Session, Depends(db_session)]) -> list[dict[str, Any]]:
    return [p.__dict__ for p in list_people(session)]


@router.get("/people/{person_id}/thumb.jpg")
def person_thumb(person_id: int, request: Request) -> Response:
    path = _thumb_path(request, person_id)
    if not path.is_file():
        raise HTTPException(404, "no thumbnail")
    return Response(
        path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"}
    )


@router.delete("/people/{person_id}")
def remove_person(
    person_id: int,
    request: Request,
    session: Annotated[Session, Depends(db_session)],
    who: Annotated[str, Depends(actor)],
) -> dict[str, Any]:
    if not delete_person(session, person_id, actor=who):
        raise HTTPException(404, "no such person")
    session.commit()
    _thumb_path(request, person_id).unlink(missing_ok=True)
    proc = request.app.state.engine_proc
    if proc is not None and proc.alive:
        proc.request("reload_gallery", timeout=15)
    return {"deleted": person_id}
