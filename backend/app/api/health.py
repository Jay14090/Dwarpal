"""Liveness/readiness endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app import __version__
from app.core.device import resolve_device
from app.db.session import check_database

router = APIRouter(tags=["system"])


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Always HTTP 200; `status` is "ok" only when every component is healthy."""
    config = request.app.state.config
    db = check_database(request.app.state.engine)
    try:
        device = resolve_device(config.settings.device)
    except (RuntimeError, ValueError) as exc:
        device = f"error: {exc}"
    healthy = db["status"] == "ok" and db["pgvector"] is not None and db["revision"] is not None
    return {
        "status": "ok" if healthy else "degraded",
        "version": __version__,
        "database": db,
        "device": device,
        "cameras": len([c for c in config.cameras.cameras if c.enabled]),
    }
