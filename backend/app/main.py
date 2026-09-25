"""FastAPI application for the Effort Estimation and Sprint Risk Predictor."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import health
from app.api.v1 import routes
from app.config import get_settings


def create_app() -> FastAPI:
    app = FastAPI(title="Effort Estimation and Sprint Risk Predictor API", version=__version__)
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
