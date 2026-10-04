"""Natural-language search (P7): /search, /search/parse, track thumbnails and clips."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from app.api.deps import actor, db_session
from app.api.privacy import serve_thumb
from app.clips import ClipError, blur_fn, cut_clip
from app.core.config import Config
from app.db.models import GlobalIdentity, Track
from app.pipeline.cache import TrackCache, cache_path
from app.privacy import PrivacyPolicy, audit
from app.search.parser import QueryParser
from app.search.retrieval import clip_prompt, search

log = logging.getLogger(__name__)
router = APIRouter(tags=["search"])
_lock = threading.Lock()


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(30, ge=1, le=200)
    now: datetime | None = None  # override "now" (replayed datasets, tests)


def _parser(request: Request) -> QueryParser:
    st = request.app.state
    with _lock:
        if getattr(st, "query_parser", None) is None:
            cfg: Config = st.config
            st.query_parser = QueryParser(cfg.settings.llm, cfg.cameras, cfg.settings.app.timezone)
    return st.query_parser


def embed_text(request: Request, text: str) -> np.ndarray | None:
    """CLIP text embedding: from the engine process if it runs (model already loaded), else a
    lazily loaded local encoder. None if CLIP is unavailable (results are then ranked by time)."""
    st = request.app.state
    if getattr(st, "text_embedder", None) is not None:
        return st.text_embedder(text)
    proc = st.engine_proc
    if proc is not None and proc.alive:
        try:
            r = proc.request("embed_text", timeout=60, texts=[text])
            if "vectors" in r:
                return np.asarray(r["vectors"][0], np.float32)
            log.warning("engine embed_text failed: %s", r.get("error"))
        except TimeoutError as exc:
            log.warning("%s", exc)
    with _lock:
        if getattr(st, "local_clip", None) is None and not getattr(st, "local_clip_failed", False):
            try:
                from app.core.device import resolve_device
                from app.pipeline.clip import ClipEncoder

                s = st.config.settings
                st.local_clip = ClipEncoder(s.clip, resolve_device(s.device), s.half_precision)
            except Exception as exc:
                log.error("CLIP unavailable for search (%s); ranking by time", exc)
                st.local_clip_failed = True
        clip = getattr(st, "local_clip", None)
        return None if clip is None else clip.embed_text([text])[0]


def run_search(
    request: Request, session: Session, query: str, limit: int, now: datetime | None
) -> dict[str, Any]:
    f, parsed_by = _parser(request).parse(query, now)
    vec = (
        embed_text(request, clip_prompt(f.free_text))
        if f.entity == "person" and f.free_text
        else None
    )
    cams = request.app.state.config.cameras.cameras
    zone_cameras: dict[str, list[str]] = {}
    for c in cams:
        for z in c.zones:
            zone_cameras.setdefault(z.name, []).append(c.id)
    out = search(session, f, vec, limit, zone_cameras)
    return {"query": query, "filter": f.dump(), "parsed_by": parsed_by, **out}


@router.post("/search")
def search_post(
    body: SearchIn, request: Request, session: Annotated[Session, Depends(db_session)]
) -> dict[str, Any]:
    return run_search(request, session, body.query, body.limit, body.now)


@router.get("/search")
def search_get(
    request: Request,
    session: Annotated[Session, Depends(db_session)],
    q: Annotated[str, Query(min_length=1, max_length=500)],
    limit: Annotated[int, Query(ge=1, le=200)] = 30,
) -> dict[str, Any]:
    return run_search(request, session, q, limit, None)


@router.post("/search/parse")
def parse_only(body: SearchIn, request: Request) -> dict[str, Any]:
    f, parsed_by = _parser(request).parse(body.query, body.now)
    return {"query": body.query, "filter": f.dump(), "parsed_by": parsed_by}


def _track(session: Session, track_id: int) -> Track:
    t = session.get(Track, track_id)
    if t is None:
        raise HTTPException(404, "track not found")
    return t


def track_role(session: Session, t: Track) -> str:
    gi = session.get(GlobalIdentity, t.global_id) if t.global_id is not None else None
    return (gi.role_state if gi else None) or t.role or "pending"


@router.get("/tracks/{track_id}/thumb.jpg")
def track_thumb(
    track_id: int,
    request: Request,
    session: Annotated[Session, Depends(db_session)],
    who: Annotated[str, Depends(actor)],
    unblur: bool = False,
) -> Response:
    """Head blurred unless the person is identified; `?unblur=true` for admins, audited."""
    t = _track(session, track_id)
    if not t.thumb_path or not Path(t.thumb_path).is_file():
        raise HTTPException(404, "no thumbnail")
    data = serve_thumb(Path(t.thumb_path).read_bytes(), track_role(session, t), request, session,
                       unblur, who, f"track:{track_id}")  # fmt: skip
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/tracks/{track_id}/clip.mp4")
def track_clip(
    track_id: int,
    request: Request,
    session: Annotated[Session, Depends(db_session)],
    who: Annotated[str, Depends(actor)],
    unblur: bool = False,
) -> FileResponse:
    """The track's time window (+-1 s) from the camera's video file. Unknown/pending people's heads are
    blurred (the track's own person too unless identified); `?unblur=true` (admin, audited) is never cached."""
    cfg: Config = request.app.state.config
    s = cfg.settings
    pol = PrivacyPolicy(s.privacy)
    t = _track(session, track_id)
    cam = next((c for c in cfg.cameras.cameras if c.id == t.camera_id), None)
    if cam is None or cam.source_type != "file" or t.start_frame is None or t.end_frame is None:
        raise HTTPException(404, "no recording for this track (only file cameras keep video)")
    src = Path(cam.source_uri)
    src = src if src.is_absolute() else cfg.root_dir / src
    if not src.is_file():
        raise HTTPException(404, "source video missing")
    role = track_role(session, t)
    blur = None
    if unblur:
        if not pol.is_admin(who):
            raise HTTPException(403, f"{who!r} is not a privacy admin")
        audit(session, who, "unblur_clip", f"track:{track_id}", role=role)
        session.commit()
    elif s.privacy.blur_unknown_faces:
        cp = cache_path(s.paths.cache_dir, cam.id)
        if not cp.is_file():
            raise HTTPException(
                409, "no cached detections for this camera, so the clip cannot be anonymised"
            )
        keep = None if pol.should_blur(role) else t.local_track_id
        blur = blur_fn(TrackCache(cp), keep)
    tag = "admin" if unblur else ("b" if blur is not None and pol.should_blur(role) else "k")
    name = f"{track_id}_{t.camera_id}_{t.start_frame}_{t.end_frame}_{tag}.mp4"
    out = s.paths.clips_dir / "tracks" / name
    if unblur:
        out = s.paths.clips_dir / "tmp" / f"{uuid.uuid4().hex}_{name}"
    if not out.is_file():
        try:
            cut_clip(src, out, t.start_frame, t.end_frame, blur, max_s=s.index.max_track_seconds + 2.0,
                     head_fraction=s.privacy.head_fraction)  # fmt: skip
        except ClipError as exc:
            raise HTTPException(500, str(exc)) from exc
    cleanup = BackgroundTask(out.unlink, missing_ok=True) if unblur else None
    return FileResponse(
        out, media_type="video/mp4", background=cleanup, headers={"Cache-Control": "no-store"}
    )
