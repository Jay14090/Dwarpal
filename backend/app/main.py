"""FastAPI entry point: `uvicorn app.main:app`."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app import __version__
from app.api import cameras, events, health, metrics, people, privacy, search, vehicles
from app.core.config import Config, get_config
from app.core.logging import setup_logging
from app.db.session import make_engine
from app.db.sync import sync_cameras
from app.pipeline.engine import EngineProcess, FrameHub
from app.privacy import retention_loop

log = logging.getLogger(__name__)


def create_app(
    config: Config | None = None,
    engine: Engine | None = None,
    start_engine: bool | None = None,
    start_retention: bool | None = None,
) -> FastAPI:
    config = config or get_config()
    setup_logging(config.settings.app.log_level)
    engine = engine or make_engine(config.settings.database)
    if start_engine is None:
        start_engine = (
            config.settings.pipeline.engine_autostart
            and os.environ.get("DWARPAL_ENGINE", "1") != "0"
        )

    if start_retention is None:
        start_retention = start_engine

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:  # cameras.yaml -> cameras table (plate reads / events reference it)
            with Session(engine) as session:
                sync_cameras(session, config.cameras)
                session.commit()
        except Exception as exc:
            log.warning("camera sync skipped: database unavailable (%s)", type(exc).__name__)
        if start_engine:
            app.state.engine_proc = EngineProcess(config, app.state.frame_hub)
            app.state.engine_proc.start()
        stop_retention = threading.Event()
        s = config.settings
        if start_retention:  # P10: delete unknown-person data older than the retention window
            threading.Thread(
                target=retention_loop,
                args=(lambda: Session(engine), s.privacy, s.paths.thumbs_dir, s.paths.clips_dir,
                      stop_retention, s.privacy.retention_interval_s),
                name="retention", daemon=True,
            ).start()  # fmt: skip
        yield
        stop_retention.set()
        if app.state.engine_proc is not None:
            app.state.engine_proc.stop()
        engine.dispose()

    app = FastAPI(title="Dwarpal", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.engine = engine
    app.state.frame_hub = FrameHub()
    app.state.engine_proc = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.settings.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(cameras.router)
    app.include_router(people.router)
    app.include_router(vehicles.router)
    app.include_router(search.router)
    app.include_router(events.router)
    app.include_router(metrics.router)
    app.include_router(privacy.router)
    return app


app = create_app()
