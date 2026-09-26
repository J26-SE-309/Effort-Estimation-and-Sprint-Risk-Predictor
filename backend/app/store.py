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
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.util.exc import CommandError
from sqlalchemy import distinct, func, inspect, select, text
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


def note_failure(error: Exception) -> None:
    """A database operation elsewhere (app.history) failed: stop trying for RETRY_SECONDS."""
    _failed(error)


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


# ------------------------------------------------------------------ reading back (the dashboard)


class DatabaseUnavailable(Exception):
    """The database cannot answer a read: the route answers 503."""


@contextmanager
def _reading():
    if not _usable():
        raise DatabaseUnavailable
    try:
        with db.SessionLocal() as session:
            yield session
    except SQLAlchemyError as error:
        _failed(error)
        raise DatabaseUnavailable from error


def _latest(records) -> dict:
    """The latest record per prediction (records ordered by created_at)."""
    return {record.prediction_id: record for record in records}


def _outcome_brief(outcome) -> dict | None:
    if outcome is None:
        return None
    return {"completed_in_sprint": outcome.completed_in_sprint, "actual_story_points": outcome.actual_story_points,
            "reopened": outcome.reopened}


def predictions_page(project_id: str, sprint_id: str | None = None, story_id: str | None = None, limit: int = 50,
                     offset: int = 0) -> tuple[list[dict], int | None]:
    """A project's predictions, newest first, each with its latest feedback and outcome; and the next offset."""
    with _reading() as session:
        query = select(PredictionRecord).where(PredictionRecord.project_id == project_id)
        if sprint_id is not None:
            query = query.where(PredictionRecord.sprint_id == sprint_id)
        if story_id is not None:
            query = query.where(PredictionRecord.story_id == story_id)
        records = session.scalars(query.order_by(PredictionRecord.created_at.desc(), PredictionRecord.id)
                                  .offset(offset).limit(limit + 1)).all()
        more, records = len(records) > limit, records[:limit]
        ids = [r.id for r in records]
        feedback = _latest(session.scalars(select(FeedbackRecord).where(FeedbackRecord.prediction_id.in_(ids))
                                           .order_by(FeedbackRecord.created_at)))
        outcomes = _latest(session.scalars(select(OutcomeRecord).where(OutcomeRecord.prediction_id.in_(ids))
                                           .order_by(OutcomeRecord.created_at)))
        page = []
        for record in records:
            shown = record.prediction
            page.append({
                "prediction_id": record.id, "created_at": record.created_at, "story_id": record.story_id,
                "sprint_id": record.sprint_id, "configuration_id": record.configuration_id,
                "model_version": record.model_version, "selection_mode": record.selection_mode,
                **{k: shown[k] for k in ("predicted_story_points", "prediction_interval", "effort_category",
                                         "spillover_probability", "sprint_risk_level", "confidence_score")},
                "confidence_level": shown.get("confidence_level", "low"),
                "feedback": feedback[record.id].decision if record.id in feedback else None,
                "outcome": _outcome_brief(outcomes.get(record.id)),
            })
    return page, (offset + limit if more else None)


def prediction_detail(prediction_id: str) -> dict | None:
    """One prediction as it was sent, the features the models saw, and its feedback and outcomes."""
    with _reading() as session:
        record = session.get(PredictionRecord, prediction_id)
        if record is None:
            return None
        feedback = session.scalars(select(FeedbackRecord).where(FeedbackRecord.prediction_id == prediction_id)
                                   .order_by(FeedbackRecord.created_at)).all()
        outcomes = session.scalars(select(OutcomeRecord).where(OutcomeRecord.prediction_id == prediction_id)
                                   .order_by(OutcomeRecord.created_at)).all()
        fields = ("id", "created_at", "prediction_id", "decision", "target", "recommendation_action",
                  "adjusted_story_points", "reason")
        return {
            "prediction_id": record.id, "project_id": record.project_id, "sprint_id": record.sprint_id,
            "created_at": record.created_at, "prediction": record.prediction, "features": record.features,
            "feedback": [{k: getattr(f, k) for k in fields} for f in feedback],
            "outcomes": [{"id": o.id, "created_at": o.created_at, "prediction_id": o.prediction_id,
                          **_outcome_brief(o)} for o in outcomes],
        }


def project_summary(project_id: str, sprint_id: str | None = None) -> dict:
    """How many predictions, of which kinds, what was decided, and how they turned out so far."""
    where = [PredictionRecord.project_id == project_id]
    if sprint_id is not None:
        where.append(PredictionRecord.sprint_id == sprint_id)
    level = PredictionRecord.prediction["sprint_risk_level"].as_string()
    with _reading() as session:
        count, stories, first, last = session.execute(select(
            func.count(), func.count(distinct(PredictionRecord.story_id)), func.min(PredictionRecord.created_at),
            func.max(PredictionRecord.created_at)).where(*where)).one()
        by_level = dict(session.execute(select(level, func.count()).where(*where).group_by(level)).all())
        by_configuration = dict(session.execute(select(PredictionRecord.configuration_id, func.count())
                                                .where(*where).group_by(PredictionRecord.configuration_id)).all())
        pinned = session.scalar(select(func.count()).where(*where, PredictionRecord.selection_mode == "pinned"))
        decisions = dict(session.execute(
            select(FeedbackRecord.decision, func.count()).join(PredictionRecord,
                                                               FeedbackRecord.prediction_id == PredictionRecord.id)
            .where(*where).group_by(FeedbackRecord.decision)).all())
        known = session.execute(
            select(OutcomeRecord, PredictionRecord.prediction).join(PredictionRecord,
                                                                    OutcomeRecord.prediction_id == PredictionRecord.id)
            .where(*where).order_by(OutcomeRecord.created_at)).all()
    latest = {outcome.prediction_id: (outcome, shown) for outcome, shown in known}
    done = [o.completed_in_sprint for o, _ in latest.values()]
    errors = [abs(shown["predicted_story_points"] - o.actual_story_points)
              for o, shown in latest.values() if o.actual_story_points is not None]
    return {
        "project_id": project_id, "sprint_id": sprint_id, "predictions": count, "stories": stories,
        "first_at": first, "last_at": last, "by_risk_level": by_level, "by_configuration": by_configuration,
        "pinned": pinned, "feedback": decisions, "outcomes": len(latest),
        "completed_share": sum(done) / len(done) if done else None,
        "mean_spillover_probability": (sum(s["spillover_probability"] for _, s in latest.values()) / len(latest)
                                       if latest else None),
        "effort_mae": sum(errors) / len(errors) if errors else None,
    }
