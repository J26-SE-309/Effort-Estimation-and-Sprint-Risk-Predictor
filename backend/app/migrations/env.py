"""Alembic environment: migrations run on the service's own engine (app.db). store.migrate() passes in the
connection that holds the migration lock; the alembic command line (run from backend/) opens its own."""

from alembic import context

from app import db
from app.tables import Base

target_metadata = Base.metadata


def _run(connection) -> None:
    # SQLite cannot alter most things in place: batch mode rebuilds the table instead (the tests use SQLite).
    context.configure(connection=connection, target_metadata=target_metadata,
                      render_as_batch=connection.dialect.name == "sqlite")
    with context.begin_transaction():
        context.run_migrations()


if (connection := context.config.attributes.get("connection")) is not None:
    _run(connection)
else:
    with db.engine.connect() as connection:
        _run(connection)
        connection.commit()
