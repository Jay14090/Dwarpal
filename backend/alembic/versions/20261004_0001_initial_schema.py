"""initial schema: section 5 tables, pgvector extension, HNSW cosine indexes

Revision ID: 0001
Revises:
Create Date: 2026-10-04 20:22:58.706108
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("actor", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column(
            "details", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False
        ),
        sa.Column(
            "ts", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_log")),
    )
    op.create_table(
        "cameras",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column(
            "zones", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False
        ),
        sa.Column("calibration", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source_type IN ('file', 'rtsp', 'webcam')", name=op.f("ck_cameras_source_type")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cameras")),
    )
    op.create_table(
        "people",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("display_name", sa.String(length=128), nullable=False),
        sa.Column("unit", sa.String(length=64), nullable=True),
        sa.Column("consent", sa.Boolean(), server_default="false", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("role IN ('resident', 'staff')", name=op.f("ck_people_person_role")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_people")),
    )
    op.create_table(
        "global_identities",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("role_state", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("person_id", sa.Integer(), nullable=True),
        sa.Column("first_seen", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "role_state IN ('pending', 'resident', 'staff', 'unknown')",
            name=op.f("ck_global_identities_role_state"),
        ),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["people.id"],
            name=op.f("fk_global_identities_person_id_people"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_global_identities")),
    )
    op.create_index(
        op.f("ix_global_identities_last_seen"), "global_identities", ["last_seen"], unique=False
    )
    op.create_index(
        op.f("ix_global_identities_person_id"), "global_identities", ["person_id"], unique=False
    )
    op.create_table(
        "person_embeddings",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("person_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("vector", Vector(512), nullable=False),
        sa.Column("quality", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('face', 'body')", name=op.f("ck_person_embeddings_embedding_kind")
        ),
        sa.ForeignKeyConstraint(
            ["person_id"],
            ["people.id"],
            name=op.f("fk_person_embeddings_person_id_people"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_person_embeddings")),
    )
    op.create_index(
        op.f("ix_person_embeddings_person_id"), "person_embeddings", ["person_id"], unique=False
    )
    op.create_index(
        "ix_person_embeddings_vector_hnsw",
        "person_embeddings",
        ["vector"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"vector": "vector_cosine_ops"},
    )
    op.create_table(
        "vehicles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("plate", sa.String(length=16), nullable=False),
        sa.Column("owner_person_id", sa.Integer(), nullable=True),
        sa.Column("vehicle_type", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["owner_person_id"],
            ["people.id"],
            name=op.f("fk_vehicles_owner_person_id_people"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_vehicles")),
        sa.UniqueConstraint("plate", name=op.f("uq_vehicles_plate")),
    )
    op.create_table(
        "plate_reads",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("camera_id", sa.String(length=64), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("plate_text", sa.String(length=16), nullable=False),
        sa.Column("raw_text", sa.String(length=32), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("vehicle_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("thumb_path", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('registered', 'likely_registered', 'unregistered')",
            name=op.f("ck_plate_reads_plate_status"),
        ),
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["cameras.id"],
            name=op.f("fk_plate_reads_camera_id_cameras"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["vehicle_id"],
            ["vehicles.id"],
            name=op.f("fk_plate_reads_vehicle_id_vehicles"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_plate_reads")),
    )
    op.create_index("ix_plate_reads_camera_ts", "plate_reads", ["camera_id", "ts"], unique=False)
    op.create_index(op.f("ix_plate_reads_plate_text"), "plate_reads", ["plate_text"], unique=False)
    op.create_table(
        "tracks",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("global_id", sa.BigInteger(), nullable=True),
        sa.Column("camera_id", sa.String(length=64), nullable=False),
        sa.Column("local_track_id", sa.Integer(), nullable=True),
        sa.Column("start_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("upper_color", sa.String(length=16), nullable=True),
        sa.Column("lower_color", sa.String(length=16), nullable=True),
        sa.Column("height_cm", sa.Float(), nullable=True),
        sa.Column("height_err_cm", sa.Float(), nullable=True),
        sa.Column("thumb_path", sa.Text(), nullable=True),
        sa.Column("quality", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["cameras.id"],
            name=op.f("fk_tracks_camera_id_cameras"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["global_id"],
            ["global_identities.id"],
            name=op.f("fk_tracks_global_id_global_identities"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tracks")),
    )
    op.create_index("ix_tracks_camera_start", "tracks", ["camera_id", "start_ts"], unique=False)
    op.create_index(op.f("ix_tracks_global_id"), "tracks", ["global_id"], unique=False)
    op.create_table(
        "events",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("rule", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("camera_id", sa.String(length=64), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("global_id", sa.BigInteger(), nullable=True),
        sa.Column("plate_read_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "payload", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False
        ),
        sa.Column("acknowledged", sa.Boolean(), server_default="false", nullable=False),
        sa.CheckConstraint(
            "severity IN ('low', 'medium', 'high', 'critical')", name=op.f("ck_events_severity")
        ),
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["cameras.id"],
            name=op.f("fk_events_camera_id_cameras"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["global_id"],
            ["global_identities.id"],
            name=op.f("fk_events_global_id_global_identities"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["plate_read_id"],
            ["plate_reads.id"],
            name=op.f("fk_events_plate_read_id_plate_reads"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_events")),
    )
    op.create_index(op.f("ix_events_ts"), "events", ["ts"], unique=False)
    op.create_table(
        "track_embeddings",
        sa.Column("track_id", sa.BigInteger(), nullable=False),
        sa.Column("clip", Vector(512), nullable=True),
        sa.Column("body", Vector(512), nullable=True),
        sa.ForeignKeyConstraint(
            ["track_id"],
            ["tracks.id"],
            name=op.f("fk_track_embeddings_track_id_tracks"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("track_id", name=op.f("pk_track_embeddings")),
    )
    op.create_index(
        "ix_track_embeddings_body_hnsw",
        "track_embeddings",
        ["body"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"body": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_track_embeddings_clip_hnsw",
        "track_embeddings",
        ["clip"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"clip": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index(
        "ix_track_embeddings_clip_hnsw",
        table_name="track_embeddings",
        postgresql_using="hnsw",
        postgresql_ops={"clip": "vector_cosine_ops"},
    )
    op.drop_index(
        "ix_track_embeddings_body_hnsw",
        table_name="track_embeddings",
        postgresql_using="hnsw",
        postgresql_ops={"body": "vector_cosine_ops"},
    )
    op.drop_table("track_embeddings")
    op.drop_index(op.f("ix_events_ts"), table_name="events")
    op.drop_table("events")
    op.drop_index(op.f("ix_tracks_global_id"), table_name="tracks")
    op.drop_index("ix_tracks_camera_start", table_name="tracks")
    op.drop_table("tracks")
    op.drop_index(op.f("ix_plate_reads_plate_text"), table_name="plate_reads")
    op.drop_index("ix_plate_reads_camera_ts", table_name="plate_reads")
    op.drop_table("plate_reads")
    op.drop_table("vehicles")
    op.drop_index(
        "ix_person_embeddings_vector_hnsw",
        table_name="person_embeddings",
        postgresql_using="hnsw",
        postgresql_ops={"vector": "vector_cosine_ops"},
    )
    op.drop_index(op.f("ix_person_embeddings_person_id"), table_name="person_embeddings")
    op.drop_table("person_embeddings")
    op.drop_index(op.f("ix_global_identities_person_id"), table_name="global_identities")
    op.drop_index(op.f("ix_global_identities_last_seen"), table_name="global_identities")
    op.drop_table("global_identities")
    op.drop_table("people")
    op.drop_table("cameras")
    op.drop_table("audit_log")
    # The vector extension is left installed: other objects in this database may use it.
