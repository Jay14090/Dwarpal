from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.main import create_app


def _client(config, url: str) -> TestClient:
    engine = create_engine(url, connect_args={"connect_timeout": 2})
    return TestClient(create_app(config=config, engine=engine, start_engine=False))


def test_health_reports_db_down(config):
    # Port 1 is never a Postgres server: connection is refused immediately.
    with _client(config, "postgresql+psycopg://x:y@127.0.0.1:1/none") as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["database"]["status"] == "down"
    assert body["version"]
    assert body["device"] in {"cuda", "mps", "cpu"}
    assert body["cameras"] == sum(c.enabled for c in config.cameras.cameras)


@pytest.mark.db
def test_health_ok_after_migrations(config, migrated_db_url):
    with _client(config, migrated_db_url) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"]["status"] == "ok"
    assert body["database"]["pgvector"]
    assert body["database"]["revision"] == "0001"
