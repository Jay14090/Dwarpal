"""Privacy endpoints (P10): audited unblur of a live stream, retention on demand, the audit log.

Thumbnail and clip unblur go through `?unblur=true` on their own endpoints (same admin check + audit).
"""

from __future__ import annotations

from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import db_session, require_admin
from app.db.models import AuditLog
from app.privacy import PrivacyPolicy, audit, blur_crop, jpeg, run_retention

router = APIRouter(tags=["privacy"])


def policy(request: Request) -> PrivacyPolicy:
    return PrivacyPolicy(request.app.state.config.settings.privacy)


def serve_thumb(data: bytes, role: str | None, request: Request, session: Session, unblur: bool,
                actor: str, target: str) -> bytes:  # fmt: skip
    """Thumbnail bytes as they may be shown: head blurred unless the person is identified, or an
    admin explicitly unblurs (audited)."""
    pol = policy(request)
    if unblur:
        if not pol.is_admin(actor):
            raise HTTPException(403, f"{actor!r} is not a privacy admin")
        audit(session, actor, "unblur_thumbnail", target, role=role)
        session.commit()
        return data
    if not pol.should_blur(role):
        return data
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, "unreadable thumbnail")
    out = jpeg(blur_crop(img, pol.cfg.head_fraction))
    if out is None:
        raise HTTPException(500, "could not encode thumbnail")
    return out


class UnblurIn(BaseModel):
    camera_id: str
    seconds: float = Field(60, gt=0, le=3600)
    reason: str = Field(min_length=3, max_length=500)


@router.post("/privacy/unblur-stream")
def unblur_stream(
    body: UnblurIn,
    request: Request,
    session: Annotated[Session, Depends(db_session)],
    who: Annotated[str, Depends(require_admin)],
) -> dict[str, Any]:
    proc = request.app.state.engine_proc
    if proc is None or not proc.alive:
        raise HTTPException(503, "the video engine is not running")
    r = proc.request("unblur", timeout=10, camera_id=body.camera_id, seconds=body.seconds)
    if "error" in r:
        raise HTTPException(404, r["error"])
    audit(
        session,
        who,
        "unblur_stream",
        f"camera:{body.camera_id}",
        seconds=r["seconds"],
        reason=body.reason,
    )
    return r


@router.post("/privacy/retention")
def retention_now(
    request: Request,
    session: Annotated[Session, Depends(db_session)],
    who: Annotated[str, Depends(require_admin)],
) -> dict[str, Any]:
    s = request.app.state.config.settings
    return run_retention(
        session, s.privacy, s.paths.thumbs_dir, s.paths.clips_dir, actor=who
    ).as_dict()


@router.get("/audit")
def audit_log(
    session: Annotated[Session, Depends(db_session)],
    _: Annotated[str, Depends(require_admin)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    action: str | None = None,
) -> list[dict[str, Any]]:
    q = select(AuditLog).order_by(AuditLog.ts.desc(), AuditLog.id.desc()).limit(limit)
    if action:
        q = q.where(AuditLog.action == action)
    return [
        {
            "id": a.id,
            "ts": a.ts.isoformat(),
            "actor": a.actor,
            "action": a.action,
            "target": a.target,
            "details": a.details,
        }
        for a in session.scalars(q)
    ]
