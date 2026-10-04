from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.core.config import DEFAULT_CONFIG_DIR, REPO_ROOT, Config, get_config, load_config


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A writable copy of the shipped config/ directory."""
    dst = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, dst)
    return dst


@pytest.fixture
def config(config_dir: Path) -> Config:
    return load_config(config_dir, env={})


def _admin_url() -> str:
    # get_config() reads .env, so this matches what `make dev` would use.
    return get_config().settings.database.url


@pytest.fixture(scope="session")
def fresh_db_url() -> Iterator[str]:
    """URL of a throwaway database on the configured server; skips if Postgres is down."""
    admin = make_url(_admin_url())
    try:
        engine = create_engine(
            admin, isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 2}
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable at {admin.render_as_string()}: {exc}")

    name = f"dwarpal_test_{uuid.uuid4().hex[:8]}"
    with engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    try:
        yield admin.set(database=name).render_as_string(hide_password=False)
    finally:
        with engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        engine.dispose()


def alembic_cfg(url: str) -> AlembicConfig:
    cfg = AlembicConfig(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["configure_logger"] = False
    return cfg


@pytest.fixture(scope="session")
def migrated_db_url(fresh_db_url: str) -> Iterator[str]:
    command.upgrade(alembic_cfg(fresh_db_url), "head")
    yield fresh_db_url
