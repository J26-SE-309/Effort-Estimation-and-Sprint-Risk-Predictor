"""Sprint history: copies of the platform's sprint records, for the team-history features

Revision ID: 0002
Revises: 0001
Created: 2026-09-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "history_sprints",
        sa.Column("project_id", sa.String(length=100), primary_key=True),
        sa.Column("sprint_id", sa.String(length=100), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("planned_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_history_sprints_source", "history_sprints", ["source"])

    op.create_table(
        "history_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("project_id", sa.String(length=100), nullable=False),
        sa.Column("sprint_id", sa.String(length=100), nullable=True),
        sa.Column("story_id", sa.String(length=200), nullable=False),
        sa.Column("issue_type", sa.String(length=50), nullable=True),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("left_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("points_at_commit", sa.Float(), nullable=True),
        sa.Column("points_at_close", sa.Float(), nullable=True),
        sa.Column("done_in_sprint", sa.Boolean(), nullable=True),
        sa.Column("spilled_over", sa.Boolean(), nullable=True),
        sa.Column("reopened", sa.Boolean(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hours_in_progress", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=20), nullable=False),
    )
    op.create_index("ix_history_items_project_id", "history_items", ["project_id"])
    op.create_index("ix_history_items_source", "history_items", ["source"])


def downgrade() -> None:
    op.drop_table("history_items")
    op.drop_table("history_sprints")
