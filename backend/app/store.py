"""Reading and writing this service's tables. Predictions never wait for, or fail because of, the database.

When the database cannot be reached, a write is skipped (the response says nothing was recorded) and the
database is not tried again for RETRY_SECONDS, so a database outage costs one connection timeout, not one per
request (NFR7: a valid degraded response rather than an error).

The database may be hosted far away (Neon in Singapore: about 60 ms a round trip), so the prediction path avoids
waiting for it: the audit log is written after the response is sent, and pins are read from a copy of the (small)
pins table refreshed every PIN_CACHE_SECONDS.
"""

import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.util.exc import CommandError
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import SQLAlchemyError

from app import db
from app.tables import FeedbackRecord, OutcomeRecord, PinnedConfiguration, PredictionRecord

RETRY_SECONDS = 30
PIN_CACHE_SECONDS = 10
MIGRATIONS = Path(__file__).parent / "migrations"
BASELINE = "0001"  # the tables as the service created them before it had migrations
MIGRATION_LOCK = 5_444_001  # a PostgreSQL advisory lock: one worker process migrates, the others wait
log = logging.getLogger(__name__)
_down_until = 0.0
_pins: tuple[float, dict[str, tuple[str, datetime]]] | None = None


def _usable() -> bool:
    return time.monotonic() >= _down_until


def _failed(error: Exception) -> None:
    global _down_until
    _down_until = time.monotonic() + RETRY_SECONDS
    log.warning("database unavailable, not recording for %s s: %s", RETRY_SECONDS, error)


def recently_down() -> bool:
    """True while a failed database operation keeps the service from trying again."""
    return not _usable()


def migrate(revision: str = "head") -> bool:
    """Bring the database to the latest migration (app/migrations), or up to `revision`. The worker processes
    start together: on PostgreSQL an advisory lock lets one migrate while the others wait and then find nothing
    to do. A database whose tables were created before the service had migrations is marked as being at
    BASELINE."""
    try:
        with db.engine.connect() as connection:
            postgres = connection.dialect.name == "postgresql"
            if postgres:
                connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": MIGRATION_LOCK})
            try:
                config = Config()
                config.set_main_option("script_location", str(MIGRATIONS))
                config.attributes["connection"] = connection
                tables = inspect(connection).get_table_names()
                if "alembic_version" not in tables and PredictionRecord.__tablename__ in tables:
                    command.stamp(config, BASELINE)
                command.upgrade(config, revision)
                connection.commit()
            finally:
                connection.rollback()
                if postgres:  # the lock belongs to the connection, which goes back to the pool
                    connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": MIGRATION_LOCK})
                    connection.commit()
        return True
    except (SQLAlchemyError, CommandError) as error:
        _failed(error)
        return False


def assign_ids(predictions: list[dict]) -> None:
    """Give every prediction the id its audit record, feedback and outcomes refer to."""
    for prediction in predictions:
        prediction["prediction_id"] = str(uuid.uuid4())


def record_predictions(project_id: str, sprint_id: str | None, predictions: list[dict], features: dict) -> bool:
    """Store the predictions with their feature snapshot (FR21); runs after the response is sent. False if not
    stored."""
    if not _usable():
        return False
    try:
        with db.SessionLocal() as session:
            session.add_all(PredictionRecord(
                id=p["prediction_id"], project_id=project_id, sprint_id=sprint_id, story_id=p["story_id"],
                configuration_id=p["configuration_id"], model_version=p["model_version"],
                selection_mode=p["selection_mode"], features=features.get(p["story_id"], {}), prediction=p)
                for p in predictions)
            session.commit()
        return True
    except SQLAlchemyError as error:
        _failed(error)
        return False


def _add(record) -> bool:
    if not _usable():
        return False
    try:
        with db.SessionLocal() as session:
            session.add(record)
            session.commit()
        return True
    except SQLAlchemyError as error:
        _failed(error)
        return False


def prediction_exists(prediction_id: str) -> bool | None:
    """True / False, or None when the database cannot say."""
    if not _usable():
        return None
    try:
        with db.SessionLocal() as session:
            return session.get(PredictionRecord, prediction_id) is not None
    except SQLAlchemyError as error:
        _failed(error)
        return None


def record_feedback(values: dict) -> tuple[str, bool]:
    record_id = str(uuid.uuid4())
    return record_id, _add(FeedbackRecord(id=record_id, **values))


def record_outcome(values: dict) -> tuple[str, bool]:
    record_id = str(uuid.uuid4())
    return record_id, _add(OutcomeRecord(id=record_id, **values))


def _pin_table() -> dict[str, tuple[str, datetime]]:
    """Every project's pin, read again at most every PIN_CACHE_SECONDS (a pin set through another worker process
    applies within that time); the last copy read while the database is unavailable."""
    global _pins
    now = time.monotonic()
    if (_pins and now - _pins[0] < PIN_CACHE_SECONDS) or not _usable():
        return _pins[1] if _pins else {}
    try:
        with db.SessionLocal() as session:
            table = {p.project_id: (p.configuration_id, p.pinned_at)
                     for p in session.scalars(select(PinnedConfiguration))}
        _pins = (now, table)
        return table
    except SQLAlchemyError as error:
        _failed(error)
        return _pins[1] if _pins else {}


def get_pin(project_id: str) -> tuple[str | None, datetime | None]:
    return _pin_table().get(project_id, (None, None))


def set_pin(project_id: str, configuration_id: str | None) -> bool:
    """Pin a configuration for the project, or remove the pin (None)."""
    global _pins
    _pins = None  # this worker reads the pins again
    if not _usable():
        return False
    try:
        with db.SessionLocal() as session:
            pin = session.get(PinnedConfiguration, project_id)
            if configuration_id is None:
                if pin:
                    session.delete(pin)
            elif pin:
                pin.configuration_id, pin.pinned_at = configuration_id, datetime.now(UTC)
            else:
                session.add(PinnedConfiguration(project_id=project_id, configuration_id=configuration_id))
            session.commit()
        return True
    except SQLAlchemyError as error:
        _failed(error)
        return False


def pinned_projects() -> list[tuple[str, str]]:
    return [(project, configuration) for project, (configuration, _) in _pin_table().items()]


def reset() -> None:
    """Forget a past outage and the copied pins (tests)."""
    global _down_until, _pins
    _down_until = 0.0
    _pins = None
