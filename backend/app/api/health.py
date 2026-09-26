from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__, db
from app.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    database: Literal["ok", "unavailable"]
    models: Literal["ok", "unavailable"] = "unavailable"
    loaded_configurations: list[str] = []


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check used by the gateway and docker compose."""
    from app.prediction import load_engine

    settings = get_settings()
    try:
        loaded = load_engine(settings.models_dir, settings.preload_models).router.loaded()
        models = "ok"
    except Exception:
        loaded, models = [], "unavailable"
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=__version__,
        database="ok" if db.database_ok() else "unavailable",
        models=models,
        loaded_configurations=loaded,
    )
