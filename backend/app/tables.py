"""This service's tables: the prediction audit log (FR21), feedback and outcomes (FR19) and pins (FR12)."""

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _now() -> datetime:
    return datetime.now(UTC)


class PredictionRecord(Base):
    """Every prediction served, with what produced it: any estimate can be traced to a model version (FR21)."""

    __tablename__ = "prediction_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    project_id: Mapped[str] = mapped_column(String(100), index=True)
    sprint_id: Mapped[str | None] = mapped_column(String(100))
    story_id: Mapped[str] = mapped_column(String(200), index=True)
    configuration_id: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(200))
    selection_mode: Mapped[str] = mapped_column(String(20))
    features: Mapped[dict] = mapped_column(JSON, doc="the feature snapshot the models saw")
    prediction: Mapped[dict] = mapped_column(JSON, doc="the response sent for this story")


class FeedbackRecord(Base):
    """A product owner's accept / adjust / reject decision (FR19)."""

    __tablename__ = "feedback_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    prediction_id: Mapped[str] = mapped_column(String(36), index=True)
    decision: Mapped[str] = mapped_column(String(10))
    target: Mapped[str] = mapped_column(String(20))
    recommendation_action: Mapped[str | None] = mapped_column(String(40))
    adjusted_story_points: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(Text)


class OutcomeRecord(Base):
    """What happened to the story in its sprint, for evaluating and retraining the models (FR19, FR20)."""

    __tablename__ = "outcome_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    prediction_id: Mapped[str] = mapped_column(String(36), index=True)
    completed_in_sprint: Mapped[bool] = mapped_column(Boolean)
    actual_story_points: Mapped[float | None] = mapped_column(Float)
    reopened: Mapped[bool] = mapped_column(Boolean, default=False)


class PinnedConfiguration(Base):
    """The configuration a product owner pinned for a project (FR12); it overrides the router."""

    __tablename__ = "pinned_configurations"

    project_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    configuration_id: Mapped[str] = mapped_column(String(100))
    pinned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
