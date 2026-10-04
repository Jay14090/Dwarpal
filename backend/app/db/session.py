"""Engine and session factory (sync SQLAlchemy 2 + psycopg 3)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import DatabaseSettings, get_config


def make_engine(db: DatabaseSettings) -> Engine:
    return create_engine(
        db.url,
        pool_size=db.pool_size,
        pool_pre_ping=True,
        connect_args={"connect_timeout": db.connect_timeout_s},
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine(get_config().settings.database)


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error."""
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database(engine: Engine) -> dict[str, str | None]:
    """Report DB reachability, pgvector version and Alembic revision. Never raises."""
    status: dict[str, str | None] = {"status": "down", "pgvector": None, "revision": None}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            status["status"] = "ok"
            status["pgvector"] = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            has_alembic = conn.execute(
                text("SELECT to_regclass('public.alembic_version') IS NOT NULL")
            ).scalar()
            if has_alembic:
                status["revision"] = conn.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar()
    except Exception as exc:  # health must report, not crash
        status["error"] = type(exc).__name__
    return status
