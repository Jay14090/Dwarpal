from __future__ import annotations

import random

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session

from app.db import models

from .conftest import alembic_cfg

pytestmark = pytest.mark.db

EXPECTED_TABLES = {
    "cameras",
    "people",
    "person_embeddings",
    "vehicles",
    "global_identities",
    "tracks",
    "track_embeddings",
    "plate_reads",
    "events",
    "audit_log",
}


def test_upgrade_creates_schema(migrated_db_url):
    engine = create_engine(migrated_db_url)
    try:
        tables = set(inspect(engine).get_table_names())
        assert tables >= EXPECTED_TABLES
        with engine.connect() as conn:
            assert conn.execute(text("SELECT 1 FROM pg_extension WHERE extname='vector'")).scalar()
            hnsw = conn.execute(
                text("SELECT count(*) FROM pg_indexes WHERE indexdef ILIKE '%USING hnsw%'")
            ).scalar()
        assert hnsw == 3
    finally:
        engine.dispose()


def test_models_match_migrations(migrated_db_url):
    # Raises if autogenerate would emit any operation (model/migration drift).
    command.check(alembic_cfg(migrated_db_url))


def test_vector_roundtrip_and_cosine_search(migrated_db_url):
    engine = create_engine(migrated_db_url)
    rng = random.Random(0)
    a = [rng.gauss(0, 1) for _ in range(512)]
    b = [rng.gauss(0, 1) for _ in range(512)]
    try:
        with Session(engine) as s:
            person = models.Person(role="resident", display_name="Test", consent=True)
            s.add(person)
            s.flush()
            s.add_all(
                [
                    models.PersonEmbedding(person_id=person.id, kind="face", vector=a),
                    models.PersonEmbedding(person_id=person.id, kind="body", vector=b),
                ]
            )
            s.flush()
            nearest = s.execute(
                select(models.PersonEmbedding.kind)
                .order_by(models.PersonEmbedding.vector.cosine_distance([x + 0.01 for x in a]))
                .limit(1)
            ).scalar_one()
            assert nearest == "face"
            s.rollback()
    finally:
        engine.dispose()


def test_enum_check_constraint_enforced(migrated_db_url):
    engine = create_engine(migrated_db_url)
    try:
        with engine.connect() as conn, pytest.raises(Exception, match="ck_people_person_role"):
            conn.execute(text("INSERT INTO people (role, display_name) VALUES ('admin', 'x')"))
    finally:
        engine.dispose()


def test_downgrade_then_upgrade(fresh_db_url, migrated_db_url):
    cfg = alembic_cfg(migrated_db_url)
    command.downgrade(cfg, "base")
    engine = create_engine(migrated_db_url)
    try:
        assert not (set(inspect(engine).get_table_names()) & EXPECTED_TABLES)
    finally:
        engine.dispose()
    command.upgrade(cfg, "head")
