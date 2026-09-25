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


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check used by the gateway and docker compose."""
    return HealthResponse(
        status="ok",
        service=get_settings().service_name,
        version=__version__,
        database="ok" if db.database_ok() else "unavailable",
    )
