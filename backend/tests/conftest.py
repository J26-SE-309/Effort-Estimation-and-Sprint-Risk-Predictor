import os

# Tests use an in-memory SQLite database instead of the component's PostgreSQL (set before the app is imported).
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import db, store  # noqa: E402
from app.main import create_app  # noqa: E402
from app.tables import Base  # noqa: E402


@pytest.fixture
def client():
    """The service with the real trained models from ml-engine/models/arena-v1 and a fresh database."""
    Base.metadata.drop_all(db.engine)
    store.reset()
    with TestClient(create_app()) as test_client:  # runs the start-up: models loaded, tables created
        yield test_client


@pytest.fixture
def client_without_database(monkeypatch):
    """The service when its database is down: predictions must still work (NFR7)."""
    monkeypatch.setattr(db, "database_ok", lambda: False)
    monkeypatch.setattr(store, "_usable", lambda: False)
    with TestClient(create_app()) as test_client:
        yield test_client
