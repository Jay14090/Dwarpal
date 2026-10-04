"""Natural-language search (P7): /search, /search/parse, track thumbnails and clips."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import db_session
from app.core.config import Config
from app.datasets.common import probe_video
from app.db.models import Track
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


@router.get("/tracks/{track_id}/thumb.jpg")
def track_thumb(track_id: int, session: Annotated[Session, Depends(db_session)]) -> Response:
    t = _track(session, track_id)
    if not t.thumb_path or not Path(t.thumb_path).is_file():
        raise HTTPException(404, "no thumbnail")
    return Response(Path(t.thumb_path).read_bytes(), media_type="image/jpeg")


@router.get("/tracks/{track_id}/clip.mp4")
def track_clip(
    track_id: int, request: Request, session: Annotated[Session, Depends(db_session)]
) -> FileResponse:
    """Cut the track's time window (+1 s padding) from the camera's video file, cached on disk."""
    cfg: Config = request.app.state.config
    t = _track(session, track_id)
    cam = next((c for c in cfg.cameras.cameras if c.id == t.camera_id), None)
    if cam is None or cam.source_type != "file" or t.start_frame is None or t.end_frame is None:
        raise HTTPException(404, "no recording for this track (only file cameras keep video)")
    out = (
        cfg.settings.paths.clips_dir
        / "tracks"
        / f"{track_id}_{t.camera_id}_{t.start_frame}_{t.end_frame}.mp4"
    )
    if not out.is_file():
        src = Path(cam.source_uri)
        src = src if src.is_absolute() else cfg.root_dir / src
        if not src.is_file() or shutil.which("ffmpeg") is None:
            raise HTTPException(404, "source video or ffmpeg missing")
        fps = probe_video(src).fps or 30.0
        start = max(0.0, t.start_frame / fps - 1.0)
        dur = min(
            (t.end_frame - t.start_frame) / fps + 2.0, cfg.settings.index.max_track_seconds + 2.0
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part.mp4")
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{dur:.2f}",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-movflags", "+faststart", str(tmp)]  # fmt: skip
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise HTTPException(500, f"ffmpeg failed: {r.stderr[-300:]}")
        tmp.replace(out)
    return FileResponse(out, media_type="video/mp4")
