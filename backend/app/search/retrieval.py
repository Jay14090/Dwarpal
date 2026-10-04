"""SearchFilter -> ranked results (CLAUDE.md section 4, P7).

People: structured filters on `tracks` (role, colours, height band, camera, zones, time overlap)
in SQL, ranked by pgvector cosine distance between the track's CLIP embedding and the CLIP text
embedding of `free_text` (or newest first when there is no free text). When colour filters leave
nothing but there is free text, the colours are relaxed and CLIP ranking alone is used; the
response says so (`relaxed`), so a wrong colour estimate does not hide a person.

Vehicles: `plate_reads` by plate (exact, then edit distance 1), camera, time and status.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import Select, and_, func, select
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Session
from sqlalchemy.types import Text

from app.anpr.registry import levenshtein
from app.db.models import GlobalIdentity, PlateRead, Track, TrackEmbedding
from app.search.parser import SearchFilter

PERSON_NOUNS = {"person", "man", "woman", "guy", "lady", "boy", "girl", "kid", "child", "people"}
VEHICLE_STATUS = {"resident": "registered", "staff": "registered", "unknown": "unregistered"}


def clip_prompt(free_text: str) -> str:
    words = set(free_text.lower().split())
    return (
        f"a cctv photo of {free_text}"
        if words & PERSON_NOUNS
        else f"a cctv photo of a person, {free_text}"
    )


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts else None


def _person_query(f: SearchFilter, colours: bool) -> Select:
    role = func.coalesce(GlobalIdentity.role_state, Track.role)
    q = (
        select(Track, role.label("role_now"), TrackEmbedding.clip)
        .outerjoin(GlobalIdentity, GlobalIdentity.id == Track.global_id)
        .outerjoin(TrackEmbedding, TrackEmbedding.track_id == Track.id)
    )
    conds = []
    if f.roles:
        conds.append(role.in_(f.roles))
    if colours and f.upper_color:
        conds.append(Track.upper_color == f.upper_color)
    if colours and f.lower_color:
        conds.append(Track.lower_color == f.lower_color)
    if f.height_cm and (f.height_cm.min is not None or f.height_cm.max is not None):
        err = func.coalesce(Track.height_err_cm, 0.0)
        conds.append(Track.height_cm.is_not(None))
        if f.height_cm.min is not None:
            conds.append(Track.height_cm + err >= f.height_cm.min)
        if f.height_cm.max is not None:
            conds.append(Track.height_cm - err <= f.height_cm.max)
    if f.cameras:
        conds.append(Track.camera_id.in_(f.cameras))
    if f.zones:
        conds.append(Track.zones.has_any(func.cast(f.zones, ARRAY(Text))))
    if f.time_range:
        if f.time_range.from_:
            conds.append(Track.end_ts >= f.time_range.from_)
        if f.time_range.to:
            conds.append(Track.start_ts <= f.time_range.to)
    return q.where(and_(*conds)) if conds else q


def search_people(
    session: Session, f: SearchFilter, text_vec: np.ndarray | None, limit: int = 30
) -> dict[str, Any]:
    relaxed: list[str] = []

    def run(colours: bool) -> list:
        q = _person_query(f, colours)
        if text_vec is not None:
            dist = TrackEmbedding.clip.cosine_distance(text_vec.tolist())
            q = q.add_columns(dist.label("dist")).order_by(
                dist.asc().nulls_last(), Track.start_ts.desc()
            )
        else:
            q = q.add_columns(func.cast(None, Text).label("dist")).order_by(Track.start_ts.desc())
        return list(session.execute(q.limit(limit)))

    rows = run(True)
    if not rows and text_vec is not None and (f.upper_color or f.lower_color):
        rows = run(False)
        relaxed = [k for k in ("upper_color", "lower_color") if getattr(f, k)]
    results = []
    for track, role_now, _clip, dist in rows:
        results.append(
            {
                "kind": "track",
                "track_id": track.id,
                "global_id": track.global_id,
                "camera_id": track.camera_id,
                "start_ts": _iso(track.start_ts),
                "end_ts": _iso(track.end_ts),
                "role": role_now or "pending",
                "upper_color": track.upper_color,
                "lower_color": track.lower_color,
                "height_cm": track.height_cm,
                "height_err_cm": track.height_err_cm,
                "zones": track.zones,
                "score": None if dist is None else round(1.0 - float(dist), 4),
                "thumb_url": f"/tracks/{track.id}/thumb.jpg" if track.thumb_path else None,
                "clip_url": f"/tracks/{track.id}/clip.mp4"
                if track.start_frame is not None
                else None,
            }
        )
    return {
        "results": results,
        "relaxed": relaxed,
        "ranked_by": "clip" if text_vec is not None else "time",
    }


def search_vehicles(
    session: Session,
    f: SearchFilter,
    limit: int = 50,
    zone_cameras: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Plate reads carry no zone; a zone narrows the search to the cameras that contain it."""
    conds = []
    cams = set(f.cameras)
    if not cams and f.zones and zone_cameras:
        cams = {c for z in f.zones for c in zone_cameras.get(z, [])}
    if cams:
        conds.append(PlateRead.camera_id.in_(sorted(cams)))
    if f.time_range:
        if f.time_range.from_:
            conds.append(PlateRead.ts >= f.time_range.from_)
        if f.time_range.to:
            conds.append(PlateRead.ts <= f.time_range.to)
    statuses = sorted({VEHICLE_STATUS[r] for r in f.roles})
    if statuses:
        st = [*statuses, "likely_registered"] if "registered" in statuses else statuses
        conds.append(PlateRead.status.in_(st))
    match = "all"
    if f.plate:
        exact = list(
            session.scalars(
                select(PlateRead)
                .where(PlateRead.plate_text == f.plate, *conds)
                .order_by(PlateRead.ts.desc())
                .limit(limit)
            )
        )
        if exact:
            reads, match = exact, "exact"
        else:  # one OCR slip either way
            texts = session.scalars(select(PlateRead.plate_text).where(*conds).distinct()).all()
            near = [t for t in texts if levenshtein(t, f.plate) <= 1]
            reads = list(
                session.scalars(
                    select(PlateRead)
                    .where(PlateRead.plate_text.in_(near), *conds)
                    .order_by(PlateRead.ts.desc())
                    .limit(limit)
                )
            ) if near else []  # fmt: skip
            match = "fuzzy"
    else:
        q = select(PlateRead).where(and_(*conds)) if conds else select(PlateRead)
        reads = list(session.scalars(q.order_by(PlateRead.ts.desc()).limit(limit)))
    return {
        "results": [
            {
                "kind": "plate_read",
                "plate_read_id": r.id,
                "plate": r.plate_text,
                "status": r.status,
                "camera_id": r.camera_id,
                "ts": _iso(r.ts),
                "confidence": round(r.confidence, 3),
                "vehicle_id": r.vehicle_id,
                "thumb_url": f"/plates/{r.id}/thumb.jpg" if r.thumb_path else None,
            }
            for r in reads
        ],
        "plate_match": match,
        "relaxed": [],
        "ranked_by": "time",
    }


def search(
    session: Session,
    f: SearchFilter,
    text_vec: np.ndarray | None = None,
    limit: int = 30,
    zone_cameras: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    if f.entity == "vehicle":
        return search_vehicles(session, f, limit, zone_cameras)
    return search_people(session, f, text_vec if f.free_text else None, limit)
