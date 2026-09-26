from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__, db, store
from app.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    database: Literal["ok", "unavailable"]
    database_location: Literal["local", "hosted"] = "local"
    models: Literal["ok", "unavailable"] = "unavailable"
    loaded_configurations: list[str] = []


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check used by the gateway and docker compose.

    A hosted database is not queried: the web app asks for the platform's health every 15 s, which would keep Neon
    from ever suspending (its free plan has 100 compute-hours a month). Its status is then the last one seen.
    """
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
        database="ok" if (not store.recently_down() if db.hosted else db.database_ok()) else "unavailable",
        database_location="hosted" if db.hosted else "local",
        models=models,
        loaded_configurations=loaded,
    )
