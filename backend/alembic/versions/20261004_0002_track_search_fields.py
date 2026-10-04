"""track search fields

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-04 21:55:53.803175
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("tracks", sa.Column("role", sa.String(length=16), nullable=True))
    op.add_column(
        "tracks",
        sa.Column(
            "zones", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False
        ),
    )
    op.add_column("tracks", sa.Column("start_frame", sa.Integer(), nullable=True))
    op.add_column("tracks", sa.Column("end_frame", sa.Integer(), nullable=True))
    op.create_index("ix_tracks_start_ts", "tracks", ["start_ts"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_tracks_start_ts", table_name="tracks")
    op.drop_column("tracks", "end_frame")
    op.drop_column("tracks", "start_frame")
    op.drop_column("tracks", "zones")
    op.drop_column("tracks", "role")
