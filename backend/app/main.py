"""FastAPI entry point: `uvicorn app.main:app`."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app import __version__
from app.api import cameras, health, people, vehicles
from app.core.config import Config, get_config
from app.core.logging import setup_logging
from app.db.session import make_engine
from app.db.sync import sync_cameras
from app.pipeline.engine import EngineProcess, FrameHub

log = logging.getLogger(__name__)


def create_app(
    config: Config | None = None, engine: Engine | None = None, start_engine: bool | None = None
) -> FastAPI:
    config = config or get_config()
    setup_logging(config.settings.app.log_level)
    engine = engine or make_engine(config.settings.database)
    if start_engine is None:
        start_engine = (
            config.settings.pipeline.engine_autostart
            and os.environ.get("DWARPAL_ENGINE", "1") != "0"
        )

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
        yield
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
    return app


app = create_app()
