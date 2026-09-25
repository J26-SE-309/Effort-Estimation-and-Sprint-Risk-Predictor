import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import create_app


@pytest.fixture
def client(monkeypatch):
    # Unit tests never need the real database.
    monkeypatch.setattr(db, "database_ok", lambda: False)
    return TestClient(create_app())
