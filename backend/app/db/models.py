"""ORM models for the schema in CLAUDE.md section 5.

Enums are VARCHAR + CHECK (`native_enum=False`) so later migrations can add values
without ALTER TYPE.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBED_DIM = 512

# Deterministic constraint names keep Alembic diffs stable.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _enum(*values: str, name: str) -> Enum:
    return Enum(*values, name=name, native_enum=False, create_constraint=True, length=32)


SourceType = _enum("file", "rtsp", "webcam", name="source_type")
PersonRole = _enum("resident", "staff", name="person_role")
EmbeddingKind = _enum("face", "body", name="embedding_kind")
RoleState = _enum("pending", "resident", "staff", "unknown", name="role_state")
PlateStatus = _enum("registered", "likely_registered", "unregistered", name="plate_status")
Severity = _enum("low", "medium", "high", "critical", name="severity")

TZ = DateTime(timezone=True)


def _now() -> Any:
    return mapped_column(TZ, server_default=func.now(), nullable=False)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    source_type: Mapped[str] = mapped_column(SourceType)
    source_uri: Mapped[str] = mapped_column(Text)
    zones: Mapped[list[Any]] = mapped_column(JSONB, server_default="[]")
    calibration: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _now()


class Person(Base):
    __tablename__ = "people"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(PersonRole)
    display_name: Mapped[str] = mapped_column(String(128))
    unit: Mapped[str | None] = mapped_column(String(64))
    consent: Mapped[bool] = mapped_column(Boolean, server_default="false")
    created_at: Mapped[datetime] = _now()


class PersonEmbedding(Base):
    __tablename__ = "person_embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("people.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(EmbeddingKind)
    vector: Mapped[Any] = mapped_column(Vector(EMBED_DIM))
    quality: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        Index(
            "ix_person_embeddings_vector_hnsw",
            "vector",
            postgresql_using="hnsw",
            postgresql_ops={"vector": "vector_cosine_ops"},
        ),
    )


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plate: Mapped[str] = mapped_column(String(16), unique=True)
    owner_person_id: Mapped[int | None] = mapped_column(
        ForeignKey("people.id", ondelete="SET NULL")
    )
    vehicle_type: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = _now()


class GlobalIdentity(Base):
    __tablename__ = "global_identities"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    role_state: Mapped[str] = mapped_column(RoleState, server_default="pending")
    person_id: Mapped[int | None] = mapped_column(
        ForeignKey("people.id", ondelete="SET NULL"), index=True
    )
    first_seen: Mapped[datetime] = mapped_column(TZ)
    last_seen: Mapped[datetime] = mapped_column(TZ, index=True)


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    global_id: Mapped[int | None] = mapped_column(
        ForeignKey("global_identities.id", ondelete="SET NULL"), index=True
    )
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    local_track_id: Mapped[int | None] = mapped_column(Integer)  # tracker ID within the camera
    start_ts: Mapped[datetime] = mapped_column(TZ)
    end_ts: Mapped[datetime] = mapped_column(TZ)
    upper_color: Mapped[str | None] = mapped_column(String(16))
    lower_color: Mapped[str | None] = mapped_column(String(16))
    height_cm: Mapped[float | None] = mapped_column(Float)
    height_err_cm: Mapped[float | None] = mapped_column(Float)
    thumb_path: Mapped[str | None] = mapped_column(Text)
    quality: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (Index("ix_tracks_camera_start", "camera_id", "start_ts"),)


class TrackEmbedding(Base):
    __tablename__ = "track_embeddings"

    track_id: Mapped[int] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True
    )
    clip: Mapped[Any | None] = mapped_column(Vector(EMBED_DIM))
    body: Mapped[Any | None] = mapped_column(Vector(EMBED_DIM))

    __table_args__ = (
        Index(
            "ix_track_embeddings_clip_hnsw",
            "clip",
            postgresql_using="hnsw",
            postgresql_ops={"clip": "vector_cosine_ops"},
        ),
        Index(
            "ix_track_embeddings_body_hnsw",
            "body",
            postgresql_using="hnsw",
            postgresql_ops={"body": "vector_cosine_ops"},
        ),
    )


class PlateRead(Base):
    __tablename__ = "plate_reads"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    ts: Mapped[datetime] = mapped_column(TZ)
    plate_text: Mapped[str] = mapped_column(String(16), index=True)
    raw_text: Mapped[str | None] = mapped_column(String(32))  # OCR output before normalization
    confidence: Mapped[float] = mapped_column(Float)
    vehicle_id: Mapped[int | None] = mapped_column(ForeignKey("vehicles.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(PlateStatus)
    thumb_path: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_plate_reads_camera_ts", "camera_id", "ts"),)


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    rule: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(Severity)
    camera_id: Mapped[str | None] = mapped_column(ForeignKey("cameras.id", ondelete="SET NULL"))
    ts: Mapped[datetime] = mapped_column(TZ, index=True)
    global_id: Mapped[int | None] = mapped_column(
        ForeignKey("global_identities.id", ondelete="SET NULL")
    )
    plate_read_id: Mapped[int | None] = mapped_column(
        ForeignKey("plate_reads.id", ondelete="SET NULL")
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    acknowledged: Mapped[bool] = mapped_column(Boolean, server_default="false")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    ts: Mapped[datetime] = _now()
