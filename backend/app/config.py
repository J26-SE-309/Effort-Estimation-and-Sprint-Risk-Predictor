"""Service settings, read from environment variables or backend/.env (see .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "effort-estimation"
    # This component's own database and account. No other component connects to it.
    database_url: str = "postgresql+psycopg://effort_user:effort-local@localhost:5444/effort_db"
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
