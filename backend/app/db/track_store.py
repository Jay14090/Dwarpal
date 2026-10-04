"""Persist finished tracks (P6): tracks + track_embeddings rows, thumbnails, global identities."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import GlobalIdentity, Track, TrackEmbedding
from app.pipeline.indexer import DbWriter, FinishedTrack


def utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, UTC)


def next_global_id(session: Session) -> int:
    return int(session.scalar(select(func.coalesce(func.max(GlobalIdentity.id), 0))) or 0) + 1


def upsert_identity(
    session: Session, gid: int, ts: float, role: str | None = None, person_id: int | None = None
) -> None:
    stmt = insert(GlobalIdentity).values(
        id=gid,
        role_state=role or "pending",
        person_id=person_id,
        first_seen=utc(ts),
        last_seen=utc(ts),
    )
    updates = {"last_seen": func.greatest(GlobalIdentity.last_seen, stmt.excluded.last_seen)}
    if role is not None:
        updates["role_state"] = stmt.excluded.role_state
        updates["person_id"] = stmt.excluded.person_id
    session.execute(stmt.on_conflict_do_update(index_elements=[GlobalIdentity.id], set_=updates))


class TrackStore:
    def __init__(
        self,
        writer: DbWriter,
        thumbs_dir: Path,
        on_stored: Callable[[int, FinishedTrack], None] | None = None,
    ) -> None:
        self.writer = writer
        self.thumbs_dir = thumbs_dir
        self.on_stored = on_stored

    def identity_created(self, gid: int, ts: float) -> None:
        self.writer.submit(lambda s: upsert_identity(s, gid, ts))

    def identity_changed(self, gid: int, role: str, person_id: int | None, ts: float) -> None:
        self.writer.submit(lambda s: upsert_identity(s, gid, ts, role, person_id))

    def store(self, ft: FinishedTrack) -> None:
        self.writer.submit(
            lambda s: self._write(s, ft),
            (lambda tid: self.on_stored(tid, ft)) if self.on_stored else None,
        )

    def _write(self, s: Session, ft: FinishedTrack) -> int:
        if ft.global_id is not None:
            upsert_identity(s, ft.global_id, ft.end_ts)
        row = Track(
            global_id=ft.global_id, camera_id=ft.camera_id, local_track_id=ft.local_track_id,
            start_ts=utc(ft.start_ts), end_ts=utc(ft.end_ts), upper_color=ft.upper_color, lower_color=ft.lower_color,
            height_cm=ft.height_cm, height_err_cm=ft.height_err_cm, quality=ft.quality, role=ft.role,
            zones=ft.zones, start_frame=ft.start_frame, end_frame=ft.end_frame,
        )  # fmt: skip
        s.add(row)
        s.flush()
        if ft.thumb_jpeg:
            path = self.thumbs_dir / "tracks" / f"{row.id}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(ft.thumb_jpeg)
            s.execute(update(Track).where(Track.id == row.id).values(thumb_path=str(path)))
        if ft.clip is not None or ft.body is not None:
            s.add(TrackEmbedding(track_id=row.id, clip=ft.clip, body=ft.body))
        return row.id
