# Database migrations with Alembic

This guide explains what a database migration is, what Alembic does, how this service uses it, and how to change
the database safely.

## What a migration is

The service's code describes the tables it expects in `backend/app/tables.py`: for example `history_sprints`, with
a `sprint_id`, a `started_at` time and so on. The database, meanwhile, holds tables as they were created some time
ago, **with real data in them**. When the code changes (a new table, a new column), the database has to be changed
to match, without losing that data.

A **migration** is a small script that makes one such change: "create the `history_sprints` table", "add a
`team_name` column", "rename `points` to `story_points`". Every migration has two directions:

- **upgrade**: make the change;
- **downgrade**: undo it.

Migrations are numbered and form a chain. Each one says which migration comes before it. So the database's
structure has a **version**, just as code has commits. It works like git for the shape of the tables (not for the
data in them).

### Why the service doesn't just create the tables at start-up

SQLAlchemy can create every table from `tables.py` (`create_all`), and the service used to do that. But
`create_all` only creates tables that are **missing**; it never changes a table that already exists. Adding a
column to `tables.py` would do nothing to Neon's existing table, and the service would then fail at its first
query. Dropping and recreating the tables would work, but it would delete everything in them. Migrations change
the table in place and keep the data.

## What Alembic is

**Alembic** is the migration tool made for SQLAlchemy, the library the service uses to talk to the database. It
keeps two things:

1. **The migration scripts**, in `backend/app/migrations/versions/`. Each file has an id (the *revision*), the
   revision before it, and `upgrade()` / `downgrade()` functions.

   | Revision | What it does |
   |---|---|
   | `0001` | The first tables: prediction audit log, feedback, outcomes, pins |
   | `0002` | Sprint history: `history_sprints`, `history_items` |

2. **A one-row table in the database**, `alembic_version`, holding the revision the database is at. Alembic
   compares it with the scripts and runs only the ones still missing.

Words you'll meet:

| Word | Meaning |
|---|---|
| revision | One migration, e.g. `0002` |
| head | The newest revision |
| upgrade | Run the missing migrations (`upgrade head`: all of them) |
| downgrade | Undo migrations (`downgrade -1`: the last one) |
| autogenerate | Alembic compares `tables.py` with a database and writes a draft migration for the difference |
| stamp | Write a revision into `alembic_version` without running anything (for adopting an existing database) |

## How this service uses it

- **At start-up, the service migrates its own database** (`store.migrate()` in `app/store.py`). It runs the
  missing migrations before the first request, on Neon or on the local `effort-db`, whichever `DATABASE_URL`
  names.
- **Several worker processes start at once.** On PostgreSQL, a lock lets one of them migrate while the others
  wait, then find nothing left to do.
- **A database created before migrations existed** already has the four tables of `0001`. The service notices
  that, marks it as being at `0001` (a *stamp*), then runs `0002` onwards.
- **`python -m app.devdata` and `python -m app.backup --restore` migrate too,** so they never write into tables
  of the wrong shape.
- **The tests guard against a forgotten migration.** `tests/test_database.py` builds a database from the
  migrations and checks it is exactly what `tables.py` describes. If you change `tables.py` without a migration,
  that test fails (on your laptop and in CI).

Run the commands below from `backend/`, with the repository's virtual environment. Alembic reads the database
address the same way the service does: `DATABASE_URL` if it is set, otherwise `backend/.env`, otherwise the local
`effort-db`.

## Changing the database, step by step

Example: add a `team_name` column to `history_sprints`.

1. **Change the table in `app/tables.py`:**

   ```python
   team_name: Mapped[str | None] = mapped_column(String(100))
   ```

2. **Point the commands at the local database, never at Neon.** Start it, then set the address for this terminal
   (PowerShell):

   ```powershell
   docker compose up -d effort-db
   $env:DATABASE_URL = "postgresql+psycopg://effort_user:effort-local@localhost:5444/effort_db"
   python -m alembic upgrade head        # the local database at the newest revision first
   ```

3. **Let Alembic draft the migration.** Use the next number as its id:

   ```powershell
   python -m alembic revision --autogenerate -m "Team name on sprints" --rev-id 0003
   ```

   This writes `app/migrations/versions/0003_team_name_on_sprints.py`.

4. **Read the draft and correct it.** Autogenerate compares shapes; it doesn't know what you meant:
   - **A renamed column shows up as "drop the old column, add a new one",** which deletes the data. Replace it
     with `op.alter_column("table", "old", new_column_name="new")`.
   - **A new column that must never be empty (`nullable=False`), on a table that already has rows,** needs a value
     for those rows: a `server_default=...`, or add it as nullable, fill it with `op.execute("UPDATE ...")`,
     then make it required.
   - **It never moves or fixes data.** Write that yourself with `op.execute(...)`.
   - **Check `downgrade()` really undoes `upgrade()`.**
   - Tidy the file in the style of `0001` and `0002`: a docstring saying what and why, double quotes, plain
     `op.create_index` calls.

5. **Test it on the local database, both directions:**

   ```powershell
   python -m alembic upgrade head
   python -m alembic downgrade -1
   python -m alembic upgrade head
   python -m pytest                     # includes the check that the migrations match tables.py
   ```

6. **Commit `tables.py` and the migration together,** in one commit.

7. **Apply it to Neon** by starting the service (it migrates at start-up). Back up Neon first:

   ```powershell
   Remove-Item Env:DATABASE_URL         # back to backend/.env, i.e. Neon
   python -m app.backup
   ```

## Rules

- **Never edit a migration that has already run on Neon.** Neon has recorded it as done and won't run it again.
  Fix things with a new migration.
- **Never delete old migrations.** A new, empty database is built by running all of them in order.
- **Don't run `alembic downgrade` against Neon.** Undoing `0002` there drops the sprint history tables *and their
  rows*. Downgrades are for your local database.
- **One change, one migration,** numbered in order (`0003`, `0004`, ...). If two people each make a `0003` on
  different branches, renumber one before merging, and set its `down_revision` to the other.
- **Tests run on SQLite,** which cannot alter most things in place. `migrations/env.py` turns on Alembic's *batch
  mode* for SQLite, which rebuilds the table instead, so the same migration works on both.

## Useful commands

| Command | What it does |
|---|---|
| `python -m alembic current` | The revision the database is at |
| `python -m alembic history` | Every migration, newest first |
| `python -m alembic upgrade head` | Run the missing migrations |
| `python -m alembic downgrade -1` | Undo the last one (local database only) |
| `python -m alembic revision --autogenerate -m "..." --rev-id 0003` | Draft a migration from the changes in `tables.py` |
| `python -m alembic stamp 0001` | Mark the database as being at `0001` without running anything (rarely needed; the service does it itself) |

## When something goes wrong

| Message | Cause | What to do |
|---|---|---|
| `Can't locate revision identified by '0003'` | The database is newer than your code, e.g. you switched to an older branch | Switch back to the branch that has `0003`, or use a local database made for this branch |
| `table ... already exists` | The tables exist but `alembic_version` doesn't, so Alembic tries to create them again | The service stamps such a database itself. From the command line: `python -m alembic stamp 0001`, then `upgrade head` |
| `Target database is not up to date` (on `revision --autogenerate`) | The database isn't at the newest revision, so the comparison would be wrong | `python -m alembic upgrade head` first |
| The service logs `database unavailable, not recording` right at start-up | The migration failed; the service keeps answering predictions but records nothing | Read the error above it in the log. PostgreSQL runs each migration in a transaction, so a failed one leaves the database as it was |
