"""Back up this service's tables to a file, and restore such a file into an empty database.

Run from backend/ (the database is the service's: DATABASE_URL in backend/.env, else the local effort-db):
    python -m app.backup                  # every table -> <Datasets>/effort-risk/backups/effort-db-<time>.jsonl.gz
    python -m app.backup --restore FILE   # into an empty database (migrated to the backup's revision, then on)

The hosted database keeps only 6 hours of history on Neon's free plan, and the feedback and outcomes are the
thesis's evaluation data: take a backup regularly. The file holds rows as JSON lines, not SQL, so it restores into
any PostgreSQL version (and SQLite) without pg_dump; its first line records the migration it was taken at.
"""

import argparse
import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

from alembic.runtime.migration import MigrationContext
from sqlalchemy import DateTime, func, select

from app import db, store
from app.tables import Base

FORMAT = 1
BATCH = 1000


def _default_directory() -> Path:
    from erp import config  # the Datasets folder next to the repositories (ERP_DATA_DIR)

    return config.DATA_DIR / "effort-risk" / "backups"


def _revision(connection) -> str | None:
    return MigrationContext.configure(connection).get_current_revision()


def backup(directory: Path) -> tuple[Path, dict[str, int]]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"effort-db-{datetime.now(UTC):%Y%m%d-%H%M%S}.jsonl.gz"
    counts: dict[str, int] = {}
    with db.engine.connect() as connection, gzip.open(path, "wt", encoding="utf-8") as out:
        out.write(json.dumps({"format": FORMAT, "revision": _revision(connection),
                              "created": datetime.now(UTC).isoformat(timespec="seconds")}) + "\n")
        for table in Base.metadata.sorted_tables:
            counts[table.name] = 0
            for row in connection.execution_options(yield_per=BATCH).execute(select(table)):
                values = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in row._mapping.items()}
                out.write(json.dumps({"table": table.name, "row": values}) + "\n")
                counts[table.name] += 1
    return path, counts


def restore(path: Path) -> dict[str, int]:
    tables = {table.name: table for table in Base.metadata.sorted_tables}
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        header = json.loads(next(lines))
        if header.get("format") != FORMAT:
            raise SystemExit(f"{path} is not a backup of this service (format {header.get('format')!r})")
        if not store.migrate(header["revision"]):  # the tables as they were when the backup was taken
            raise SystemExit("the database cannot be reached or migrated")
        counts = dict.fromkeys(tables, 0)
        with db.engine.begin() as connection:
            if (revision := _revision(connection)) != header["revision"]:
                raise SystemExit(f"the backup was taken at migration {header['revision']}, the database is already "
                                 f"at {revision}: restore into a new, empty database")
            filled = [name for name, table in tables.items()
                      if connection.execute(select(func.count()).select_from(table)).scalar()]
            if filled:
                raise SystemExit(f"restore only into an empty database; these tables have rows: {', '.join(filled)}")
            pending: dict[str, list[dict]] = {name: [] for name in tables}
            for line in lines:
                item = json.loads(line)
                table = tables[item["table"]]
                pending[table.name].append({
                    k: datetime.fromisoformat(v) if v is not None and isinstance(table.c[k].type, DateTime) else v
                    for k, v in item["row"].items()})
                counts[table.name] += 1
                if len(pending[table.name]) >= BATCH:
                    connection.execute(table.insert(), pending[table.name])
                    pending[table.name] = []
            for name, rows in pending.items():
                if rows:
                    connection.execute(tables[name].insert(), rows)
    store.migrate()  # then on to the latest migration, with the rows
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="folder for the backup (default: <Datasets>/effort-risk/backups)")
    parser.add_argument("--restore", type=Path, metavar="FILE", help="restore this backup into an empty database")
    args = parser.parse_args()
    where = "hosted" if db.hosted else "local"
    if args.restore:
        counts = restore(args.restore)
        print(f"Restored {args.restore} into the {where} database: {counts}")
    else:
        path, counts = backup(args.out or _default_directory())
        print(f"Backed up the {where} database to {path}: {counts}")


if __name__ == "__main__":
    main()
