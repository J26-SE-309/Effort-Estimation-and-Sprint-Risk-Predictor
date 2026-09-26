"""Service settings, read from environment variables or backend/.env (see .env.example)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# In the repository the trained models sit next to the service: ml-engine/models/arena-v1.
REPO_MODELS = Path(__file__).resolve().parents[2] / "ml-engine" / "models" / "arena-v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "effort-estimation"
    # This component's own database and account. No other component connects to it.
    database_url: str = "postgresql+psycopg://effort_user:effort-local@localhost:5444/effort_db"
    cors_origins: list[str] = ["http://localhost:3000"]
    # The arena's trained models (leaderboard.json and one folder per configuration).
    models_dir: Path = REPO_MODELS
    # Load the router's pooled winner at start-up, so the first request does not wait for it.
    preload_models: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
