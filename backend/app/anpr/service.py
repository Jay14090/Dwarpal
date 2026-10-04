"""Turns plate decisions into registry matches, stored plate reads and vehicle events."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.anpr.hook import PlateDecision
from app.anpr.registry import Registry
from app.db.models import PlateRead, Vehicle
from app.events import EventBus
from app.pipeline.indexer import DbWriter

log = logging.getLogger(__name__)
SHORT = {"registered": "reg", "likely_registered": "verify", "unregistered": "UNREG"}


def load_registry(session: Session, max_distance: int) -> Registry:
    return Registry(
        {plate: vid for vid, plate in session.execute(select(Vehicle.id, Vehicle.plate))},
        max_distance,
    )


class PlateService:
    def __init__(
        self,
        registry: Registry,
        thumbs_dir: Path,
        bus: EventBus | None = None,
        writer: DbWriter | None = None,
        publish: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.registry = registry
        self.thumbs_dir = thumbs_dir
        self.bus = bus
        self.writer = writer
        self.publish = publish
        self.recent: deque[dict[str, Any]] = deque(maxlen=200)
        self._lock = threading.Lock()

    def set_registry(self, registry: Registry) -> None:
        with self._lock:
            self.registry = registry

    def handle(self, d: PlateDecision) -> str:
        with self._lock:
            m = self.registry.match(d.vote.plate)
        thumb_path = None
        if d.thumb_jpeg:
            thumb_path = self.thumbs_dir / "plates" / f"{d.camera_id}_{d.frame}_{d.track_id}.jpg"
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            thumb_path.write_bytes(d.thumb_jpeg)
        info = {
            "camera_id": d.camera_id,
            "ts": d.ts,
            "plate": d.vote.plate,
            "status": m.status,
            "matched_plate": m.matched_plate,
            "vehicle_id": m.vehicle_id,
            "confidence": round(d.vote.confidence, 3),
            "reads": d.vote.reads,
            "vehicle_label": d.vehicle_label,
            "thumb_path": str(thumb_path) if thumb_path else None,
        }
        log.info("%s: plate %s -> %s (%d reads)", d.camera_id, d.vote.plate, m.status, d.vote.reads)
        if self.writer is None:
            self._after_store(info, None)
        else:
            self.writer.submit(
                lambda s: self._store(s, d, m.status, m.vehicle_id, thumb_path),
                lambda rid: self._after_store(info, rid),
            )
        return f"{d.vote.plate} {SHORT[m.status]}"

    @staticmethod
    def _store(
        s: Session, d: PlateDecision, status: str, vehicle_id: int | None, thumb: Path | None
    ) -> int:
        row = PlateRead(
            camera_id=d.camera_id,
            ts=datetime.fromtimestamp(d.ts, UTC),
            plate_text=d.vote.plate,
            raw_text=(d.vote.raw_examples or [None])[0],
            confidence=d.vote.confidence,
            vehicle_id=vehicle_id,
            status=status,
            thumb_path=str(thumb) if thumb else None,
        )
        s.add(row)
        s.flush()
        return row.id

    def _after_store(self, info: dict[str, Any], plate_read_id: int | None) -> None:
        info = {**info, "id": plate_read_id}
        self.recent.appendleft(info)
        if self.publish is not None:
            self.publish("plate", info)
        if self.bus is not None and info["status"] == "unregistered":
            self.bus.emit(
                "unregistered_vehicle", info["plate"], camera_id=info["camera_id"], ts=info["ts"],
                plate_read_id=plate_read_id,
                payload={k: info[k] for k in ("plate", "vehicle_label", "confidence", "thumb_path")},
            )  # fmt: skip
