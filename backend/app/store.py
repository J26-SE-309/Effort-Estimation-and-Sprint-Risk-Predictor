"""Reading and writing this service's tables. Predictions never wait for, or fail because of, the database.

When the database cannot be reached, a write is skipped (the response says nothing was recorded) and the
database is not tried again for RETRY_SECONDS, so a database outage costs one connection timeout, not one per
request (NFR7: a valid degraded response rather than an error).
"""

import logging
import time
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app import db
from app.tables import Base, FeedbackRecord, OutcomeRecord, PinnedConfiguration, PredictionRecord

RETRY_SECONDS = 30
log = logging.getLogger(__name__)
_down_until = 0.0


def _usable() -> bool:
    return time.monotonic() >= _down_until


def _failed(error: Exception) -> None:
    global _down_until
    _down_until = time.monotonic() + RETRY_SECONDS
    log.warning("database unavailable, not recording for %s s: %s", RETRY_SECONDS, error)


def create_tables() -> bool:
    """Create missing tables. Several worker processes start together and may race; the second try wins."""
    for attempt in range(2):
        try:
            Base.metadata.create_all(db.engine)
            return True
        except SQLAlchemyError as error:
            if attempt:
                _failed(error)
            time.sleep(0.5)
    return False


def record_predictions(project_id: str, sprint_id: str | None, predictions: list[dict], features: dict) -> bool:
    """Give every prediction an id and store it with its feature snapshot (FR21). False if not stored."""
    for prediction in predictions:
        prediction["prediction_id"] = str(uuid.uuid4())
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


def get_pin(project_id: str) -> tuple[str | None, datetime | None]:
    if not _usable():
        return None, None
    try:
        with db.SessionLocal() as session:
            pin = session.get(PinnedConfiguration, project_id)
            return (pin.configuration_id, pin.pinned_at) if pin else (None, None)
    except SQLAlchemyError as error:
        _failed(error)
        return None, None


def set_pin(project_id: str, configuration_id: str | None) -> bool:
    """Pin a configuration for the project, or remove the pin (None)."""
    if not _usable():
        return False
    try:
        with db.SessionLocal() as session:
            pin = session.get(PinnedConfiguration, project_id)
            if configuration_id is None:
                if pin:
                    session.delete(pin)
            elif pin:
                pin.configuration_id = configuration_id
            else:
                session.add(PinnedConfiguration(project_id=project_id, configuration_id=configuration_id))
            session.commit()
        return True
    except SQLAlchemyError as error:
        _failed(error)
        return False


def pinned_projects() -> list[tuple[str, str]]:
    if not _usable():
        return []
    try:
        with db.SessionLocal() as session:
            return [(p.project_id, p.configuration_id) for p in session.scalars(select(PinnedConfiguration))]
    except SQLAlchemyError as error:
        _failed(error)
        return []


def reset() -> None:
    """Forget a past outage (tests)."""
    global _down_until
    _down_until = 0.0
