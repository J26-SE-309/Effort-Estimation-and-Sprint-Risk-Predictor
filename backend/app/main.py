"""FastAPI application for the Effort Estimation and Sprint Risk Predictor."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__, db, store
from app.api import health
from app.api.v1 import routes
from app.config import get_settings
from app.prediction import get_engine

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models once, before the first request (NFR1), and migrate the database if it is up."""
    try:
        get_engine()
    except Exception as error:  # the service still starts; /estimate answers 503 until models are present
        log.error("prediction engine not loaded: %s", error)
    if db.database_ok():
        store.migrate()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Effort Estimation and Sprint Risk Predictor API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(routes.router, prefix="/api/v1")
    return app


app = create_app()
