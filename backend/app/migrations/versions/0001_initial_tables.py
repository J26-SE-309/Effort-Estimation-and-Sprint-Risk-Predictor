"""The service's tables: prediction audit log, feedback, outcomes and pins

Revision ID: 0001
Revises:
Created: 2026-09-26

The tables as the service created them before it had migrations: store.migrate() marks such a database as being
at this revision instead of creating them again.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "prediction_records",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("project_id", sa.String(length=100), nullable=False),
        sa.Column("sprint_id", sa.String(length=100), nullable=True),
        sa.Column("story_id", sa.String(length=200), nullable=False),
        sa.Column("configuration_id", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=200), nullable=False),
        sa.Column("selection_mode", sa.String(length=20), nullable=False),
        sa.Column("features", sa.JSON(), nullable=False),
        sa.Column("prediction", sa.JSON(), nullable=False),
    )
    op.create_index("ix_prediction_records_project_id", "prediction_records", ["project_id"])
    op.create_index("ix_prediction_records_story_id", "prediction_records", ["story_id"])

    op.create_table(
        "feedback_records",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prediction_id", sa.String(length=36), nullable=False),
        sa.Column("decision", sa.String(length=10), nullable=False),
        sa.Column("target", sa.String(length=20), nullable=False),
        sa.Column("recommendation_action", sa.String(length=40), nullable=True),
        sa.Column("adjusted_story_points", sa.Float(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
    )
    op.create_index("ix_feedback_records_prediction_id", "feedback_records", ["prediction_id"])

    op.create_table(
        "outcome_records",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prediction_id", sa.String(length=36), nullable=False),
        sa.Column("completed_in_sprint", sa.Boolean(), nullable=False),
        sa.Column("actual_story_points", sa.Float(), nullable=True),
        sa.Column("reopened", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_outcome_records_prediction_id", "outcome_records", ["prediction_id"])

    op.create_table(
        "pinned_configurations",
        sa.Column("project_id", sa.String(length=100), primary_key=True),
        sa.Column("configuration_id", sa.String(length=100), nullable=False),
        sa.Column("pinned_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("pinned_configurations")
    op.drop_table("outcome_records")
    op.drop_table("feedback_records")
    op.drop_table("prediction_records")
