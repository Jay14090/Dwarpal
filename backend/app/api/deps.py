"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.pipeline.engine import EngineProcess


def db_session(request: Request) -> Iterator[Session]:
    session = Session(request.app.state.engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def engine_proc(request: Request) -> EngineProcess:
    proc = request.app.state.engine_proc
    if proc is None or not proc.alive:
        raise HTTPException(503, "the video engine is not running")
    return proc


def actor(x_actor: str | None = Header(default=None)) -> str:
    """Who is acting, for the audit log. Real authentication is out of scope for the demo."""
    return (x_actor or "operator")[:64]
