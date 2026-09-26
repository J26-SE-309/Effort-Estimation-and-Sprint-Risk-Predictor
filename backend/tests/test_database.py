"""The database around the predictions: settings, migrations, backups, and what reaches it when."""

import gzip
import json

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, func, select

from app import backup, db, store
from app.config import Settings
from app.tables import Base, FeedbackRecord, OutcomeRecord, PinnedConfiguration, PredictionRecord


def _story(story_id: str) -> dict:
    return {"story_id": story_id, "title": "As a user I want to export reports so that I can share them",
            "story_points": 3}


def test_hosted_connection_strings_work_as_given():
    hosted = Settings(database_url="postgresql://owner:secret@ep-x.ap-southeast-1.aws.neon.tech/neondb?sslmode=require")
    assert hosted.database.startswith("postgresql+psycopg://owner:secret@ep-x.")
    assert Settings(database_url="postgres://u:p@h/d").database == "postgresql+psycopg://u:p@h/d"
    assert Settings(database_url=None).database == Settings().local_database_url  # no DATABASE_URL: effort-db
    assert db.is_hosted(hosted.database)
    assert not db.is_hosted("postgresql+psycopg://u:p@effort-db:5432/effort_db")
    assert not db.is_hosted("postgresql+psycopg://u:p@localhost:5444/effort_db")
    assert not db.is_hosted("sqlite+pysqlite:///:memory:")


@pytest.fixture
def fresh_database(tmp_path, monkeypatch):
    """An empty database of its own, as the service's engine."""
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'fresh.db'}")
    monkeypatch.setattr(db, "engine", engine)
    store.reset()
    yield engine
    engine.dispose()


def test_migrations_build_exactly_the_tables_the_service_uses(fresh_database):
    assert store.migrate() and store.migrate()  # the second start finds nothing to do
    head = ScriptDirectory(str(store.MIGRATIONS)).get_current_head()
    with fresh_database.connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == head
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []


def test_tables_created_before_migrations_are_adopted_not_created_again(fresh_database):
    # How the service made its tables before it had migrations: the four of the first revision.
    original = [PredictionRecord, FeedbackRecord, OutcomeRecord, PinnedConfiguration]
    Base.metadata.create_all(fresh_database, tables=[table.__table__ for table in original])
    assert store.migrate()  # adopted as the first revision, then migrated on
    with fresh_database.connect() as connection:
        head = ScriptDirectory(str(store.MIGRATIONS)).get_current_head()
        assert MigrationContext.configure(connection).get_current_revision() == head
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []


def test_the_audit_record_is_written_after_the_response(client):
    body = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": [_story("S-1")]}).json()
    prediction_id = body["predictions"][0]["prediction_id"]
    assert store.prediction_exists(prediction_id)  # the test client runs the background task before returning
    assert client.post("/api/v1/feedback", json={"prediction_id": prediction_id, "decision": "accept"}
                       ).status_code == 201


def test_backup_and_restore_give_back_every_row(client, tmp_path):
    body = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": [_story("S-1"), _story("S-2")]})
    ids = {p["prediction_id"] for p in body.json()["predictions"]}
    client.post("/api/v1/feedback", json={"prediction_id": next(iter(ids)), "decision": "reject", "reason": "big"})
    client.put("/api/v1/projects/SYN/pin", json={"configuration_id": "tfidf-rf"})

    path, counts = backup.backup(tmp_path)
    assert counts == {"prediction_records": 2, "feedback_records": 1, "outcome_records": 0,
                      "pinned_configurations": 1, "history_sprints": 0, "history_items": 0}
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        assert json.loads(next(lines))["revision"] == ScriptDirectory(str(store.MIGRATIONS)).get_current_head()

    with pytest.raises(SystemExit, match="empty database"):
        backup.restore(path)  # never on top of existing rows
    Base.metadata.drop_all(db.engine)
    with db.engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE alembic_version")
    assert backup.restore(path) == counts
    with db.SessionLocal() as session:
        assert set(session.scalars(select(PredictionRecord.id))) == ids
        assert session.scalar(select(FeedbackRecord.reason)) == "big"
        assert session.get(PinnedConfiguration, "SYN").pinned_at.year >= 2026
        assert session.scalar(select(func.count()).select_from(PredictionRecord)) == 2


def test_health_does_not_wake_a_hosted_database(client, monkeypatch):
    monkeypatch.setattr(db, "hosted", True)
    monkeypatch.setattr(db, "database_ok", lambda: pytest.fail("health queried the hosted database"))
    body = client.get("/health").json()
    assert (body["database"], body["database_location"]) == ("ok", "hosted")
    monkeypatch.setattr(store, "_usable", lambda: False)  # after a failed write
    assert client.get("/health").json()["database"] == "unavailable"


def test_a_pin_applies_at_once_in_the_worker_that_set_it(client):
    story = {"project_id": "SYN", "stories": [_story("S-1")]}
    assert client.post("/api/v1/estimate", json=story).json()["selection_mode"] == "auto"
    client.put("/api/v1/projects/SYN/pin", json={"configuration_id": "tfidf-rf"})
    assert client.post("/api/v1/estimate", json=story).json()["configuration_id"] == "tfidf-rf"
    client.delete("/api/v1/projects/SYN/pin")
    assert client.post("/api/v1/estimate", json=story).json()["selection_mode"] == "auto"
