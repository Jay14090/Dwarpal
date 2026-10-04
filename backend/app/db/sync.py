"""Mirror config/cameras.yaml into the cameras table (plate reads, tracks and events reference it)."""

from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.config import CamerasConfig
from app.db.models import Camera


def sync_cameras(session: Session, cameras: CamerasConfig) -> int:
    for cam in cameras.cameras:
        values = {
            "id": cam.id,
            "name": cam.name,
            "source_type": cam.source_type,
            "source_uri": cam.source_uri,
            "zones": [z.model_dump() for z in cam.zones],
            "calibration": cam.calibration,
        }
        stmt = insert(Camera).values(**values)
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=[Camera.id], set_={k: stmt.excluded[k] for k in values if k != "id"}
            )
        )
    return len(cameras.cameras)
