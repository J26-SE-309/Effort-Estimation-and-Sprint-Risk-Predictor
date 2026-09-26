"""Access to this component's own PostgreSQL database: hosted (Neon) or local (the effort-db container), SQLite in
the tests.

A hosted database may be asleep (Neon suspends after 5 idle minutes and closes the pool's connections): the first
connection waits for it to wake, and pool_pre_ping replaces connections it closed.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine, make_url, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings


class Base(DeclarativeBase):
    """Base class for this service's tables."""


def _engine(url: str):
    if url.startswith("sqlite") and ":memory:" in url:  # the tests: one in-memory database, one connection
        return create_engine(url, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    if url.startswith("sqlite"):  # a local file: a connection per thread, never one shared between threads
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})


def is_hosted(url: str) -> bool:
    """True for a database elsewhere (Neon), False for SQLite or this component's own container."""
    parsed = make_url(url)
    return not parsed.drivername.startswith("sqlite") and parsed.host not in LOCAL_HOSTS


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "effort-db"}
engine = _engine(get_settings().database)
hosted = is_hosted(get_settings().database)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one database session per request."""
    with SessionLocal() as session:
        yield session


def database_ok() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
