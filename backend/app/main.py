"""FastAPI entry point: `uvicorn app.main:app`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine

from app import __version__
from app.api import health
from app.core.config import Config, get_config
from app.core.logging import setup_logging
from app.db.session import make_engine


def create_app(config: Config | None = None, engine: Engine | None = None) -> FastAPI:
    config = config or get_config()
    setup_logging(config.settings.app.log_level)
    engine = engine or make_engine(config.settings.database)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        engine.dispose()

    app = FastAPI(title="Dwarpal", version=__version__, lifespan=lifespan)
    app.state.config = config
    app.state.engine = engine
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.settings.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    return app


app = create_app()
