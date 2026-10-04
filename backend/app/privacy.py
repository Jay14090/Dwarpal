"""Privacy by design (CLAUDE.md section 4, P10).

- Face blur: the head region of every person whose role is in `privacy.blur_roles` (default unknown and
  pending, i.e. anyone not identified as a consented, enrolled person) is blurred in streams, thumbnails
  and clips. The head region is a fixed fraction of the person box, so it also works at CCTV distance
  where no face detector fires.
- Thumbnails are stored once and blurred when served, from the identity's *current* role (enrolling
  someone later unblurs their history; nobody's face is shown while they are unknown or pending).
- Unblur is an admin action (actor in `privacy.admin_actors`), time-limited for live streams and
  always written to `audit_log`; unblurred clips are never cached.
- Retention: embeddings, thumbnails and clips of unknown/pending tracks (and their events' thumbnails)
  older than `privacy.unknown_retention_days` are deleted; the run is audited.
- Consent: enrollment without consent is refused (gallery_store.ConsentRequired) and the gallery only
  loads people with consent=true.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import PrivacySettings
from app.db.models import AuditLog, Event, GlobalIdentity, Track, TrackEmbedding
from app.pipeline.frames import Track as LiveTrack

log = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


def head_box(
    xyxy: Box, shape: tuple[int, ...], frac: float = 0.24, pad: float = 0.12
) -> tuple[int, int, int, int]:
    """Top `frac` of a person box, padded sideways and upwards (heads stick out of tight boxes)."""
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    H, W = shape[:2]
    return (
        max(0, int(x1 - pad * w)),
        max(0, int(y1 - 0.05 * h)),
        min(W, int(np.ceil(x2 + pad * w))),
        min(H, int(np.ceil(y1 + frac * h))),
    )


def blur_box(image: np.ndarray, box: tuple[int, int, int, int]) -> None:
    """Irreversible in-place anonymisation: pixelate, then blur (no readable detail survives)."""
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    roi[:] = cv2.GaussianBlur(
        cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), max(1.0, w / 12)
    )


def blur_heads(image: np.ndarray, boxes: Iterable[Box], frac: float = 0.24) -> np.ndarray:
    """Copy of `image` with the head region of each person box blurred."""
    out = image.copy()
    for b in boxes:
        blur_box(out, head_box(b, out.shape, frac))
    return out


def blur_crop(crop: np.ndarray, frac: float = 0.24) -> np.ndarray:
    """A person crop (box == crop) with its head blurred."""
    h, w = crop.shape[:2]
    return blur_heads(crop, [(0.0, 0.0, float(w), float(h))], frac)


@dataclass
class PrivacyPolicy:
    cfg: PrivacySettings

    def should_blur(self, role: str | None) -> bool:
        return self.cfg.blur_unknown_faces and (role or "pending") in self.cfg.blur_roles

    def is_admin(self, actor: str) -> bool:
        return actor in self.cfg.admin_actors

    def frame(self, image: np.ndarray, tracks: list[LiveTrack]) -> np.ndarray:
        boxes = [t.xyxy for t in tracks if t.is_person and self.should_blur(t.role)]
        return blur_heads(image, boxes, self.cfg.head_fraction) if boxes else image

    def crop(self, crop: np.ndarray, role: str | None) -> np.ndarray:
        return blur_crop(crop, self.cfg.head_fraction) if self.should_blur(role) else crop


def audit(session: Session, actor: str, action: str, target: str, **details: object) -> None:
    session.add(AuditLog(actor=actor, action=action, target=target, details=details))


def jpeg(image: np.ndarray, quality: int = 85) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


# --------------------------------------------------------------------------- retention


@dataclass
class RetentionReport:
    cutoff: str
    tracks: int = 0
    embeddings: int = 0
    files: int = 0
    events: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _unlink(paths: Iterable[Path]) -> int:
    n = 0
    for p in paths:
        try:
            p.unlink()
            n += 1
        except FileNotFoundError:
            pass
    return n


def run_retention(
    session: Session,
    cfg: PrivacySettings,
    thumbs_dir: Path,
    clips_dir: Path,
    now: datetime | None = None,
    actor: str = "retention-job",
) -> RetentionReport:
    """Delete embeddings, thumbnails and clips of unknown/pending tracks older than the retention window.

    The track rows themselves (camera, time, colours: no biometric data) stay, so event history and
    counts remain consistent; the identity can no longer be searched by appearance or looked at.
    """
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(days=cfg.unknown_retention_days)
    rep = RetentionReport(cutoff.isoformat())
    role = func.coalesce(GlobalIdentity.role_state, Track.role, "pending")
    q = (
        select(Track.id, Track.thumb_path)
        .outerjoin(GlobalIdentity, GlobalIdentity.id == Track.global_id)
        .where(Track.end_ts < cutoff, role.in_(["unknown", "pending"]))
    )
    rows = session.execute(q).all()
    ids = [r[0] for r in rows]
    for chunk in (ids[i : i + 1000] for i in range(0, len(ids), 1000)):
        rep.embeddings += (
            session.query(TrackEmbedding)
            .filter(TrackEmbedding.track_id.in_(chunk))
            .delete(synchronize_session=False)
        )
        session.execute(
            update(Track)
            .where(Track.id.in_(chunk), Track.thumb_path.is_not(None))
            .values(thumb_path=None)
        )
    files: list[Path] = []
    for tid, thumb in rows:
        if thumb:
            files.append(Path(thumb))
        files.extend((clips_dir / "tracks").glob(f"{tid}_*.mp4"))
    rep.tracks = len(ids)
    # events about unknown people: drop the person crop, keep the alert record
    evs = session.scalars(
        select(Event)
        .outerjoin(GlobalIdentity, GlobalIdentity.id == Event.global_id)
        .where(
            Event.ts < cutoff,
            func.coalesce(GlobalIdentity.role_state, "pending").in_(["unknown", "pending"]),
        )
    ).all()
    for e in evs:
        if e.payload.get("thumb"):
            p = thumbs_dir / "events" / f"{e.id}.jpg"
            files.append(p)
            e.payload = {**e.payload, "thumb": False}
            rep.events += 1
    rep.files = _unlink(files)
    audit(session, actor, "retention", f"unknown<{cutoff.date().isoformat()}", **rep.as_dict())
    session.commit()
    log.info("retention: %s", rep.as_dict())
    return rep


def retention_loop(
    make_session: Callable[[], Session],
    cfg: PrivacySettings,
    thumbs_dir: Path,
    clips_dir: Path,
    stop: threading.Event,
    interval_s: float,
) -> None:
    """Run retention now and then every `interval_s` until `stop` (threading.Event) is set."""
    while not stop.is_set():
        try:
            with make_session() as session:
                run_retention(session, cfg, thumbs_dir, clips_dir)
        except Exception as exc:
            log.warning("retention run failed (%s)", exc)
        t0 = time.monotonic()
        while not stop.is_set() and time.monotonic() - t0 < interval_s:
            stop.wait(min(5.0, interval_s))
