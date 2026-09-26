"""Service settings, read from environment variables or backend/.env (see .env.example)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# In the repository the trained models sit next to the service: ml-engine/models/arena-v1.
REPO_MODELS = Path(__file__).resolve().parents[2] / "ml-engine" / "models" / "arena-v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "effort-estimation"
    # This component's own database; no other component connects to it. DATABASE_URL (in backend/.env, never
    # committed) names the hosted database (Neon); without it the service uses the local one, the effort-db
    # container of docker-compose.yml.
    database_url: str | None = None
    local_database_url: str = "postgresql+psycopg://effort_user:effort-local@localhost:5444/effort_db"
    cors_origins: list[str] = ["http://localhost:3000"]
    # The arena's trained models (leaderboard.json and one folder per configuration).
    models_dir: Path = REPO_MODELS
    # Load the router's pooled winner at start-up, so the first request does not wait for it.
    preload_models: bool = True

    @property
    def database(self) -> str:
        """The database to use, with the driver SQLAlchemy needs (hosted providers hand out postgresql:// URLs)."""
        url = self.database_url or self.local_database_url
        for scheme in ("postgresql://", "postgres://"):
            if url.startswith(scheme):
                return "postgresql+psycopg://" + url.removeprefix(scheme)
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
