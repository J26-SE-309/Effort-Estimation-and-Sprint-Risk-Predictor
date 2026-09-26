"""Team sprint history (FR5): the sprint records the service keeps, the team context the models see, and the
CSV import.

The platform owns the sprints. Until it serves them through an API, the service keeps copies in its own tables
(history_sprints, history_items), each row labelled with where it came from (SOURCES). SOURCE is the one place
that says where records come from: a client of the platform's API that returns the same record format
(erp.serving.history) replaces StoredHistory, and nothing else changes. The team numbers themselves are computed
by erp.serving.history with the training pipeline's own code.

The database may be a network away (Neon: reading a long history takes seconds), so each worker keeps a prepared
copy of a project's records: after REFRESH_SECONDS the next request still answers from it while the records are
read again in the background. Only a project's first request waits for them, and an estimate falls back to a
cold start when its history cannot be read (NFR7).
"""

import io
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import SQLAlchemyError

from app import db, store
from app.tables import HistoryItem, HistorySprint, OutcomeRecord, PredictionRecord

# platform: sprint by sprint from the platform (PUT .../sprints/{id}); imported: a CSV through the API; tawos: real
# TAWOS sprints (development data); synthetic: made up for tests and demos, never used in any evaluation
SOURCES = ("platform", "imported", "tawos", "synthetic")
REFRESH_SECONDS = 60
MAX_ROWS = 200_000
BATCH = 5_000

# The CSV import: one row per story per sprint. A row without a sprint_id is a resolved story never in a sprint
# (only its cycle time counts). Times are ISO 8601; without a time zone they are taken as UTC.
CSV_COLUMNS = {
    "sprint_id": "the sprint's id in the platform (empty: a story never in a sprint)",
    "sprint_name": "the sprint's name",
    "sprint_started_at": "when the sprint started",
    "sprint_planned_end": "when it was planned to end",
    "sprint_closed_at": "when it was closed (empty while it runs)",
    "story_id": "the story's id in the platform",
    "issue_type": "Story, Task, Bug, Improvement or New Feature (others do not count for the cycle time)",
    "committed_at": "when the story was committed to the sprint (its start for a planned story)",
    "left_at": "when it was taken out of the sprint before the end (empty: still in at the end)",
    "points_at_commit": "its story points when committed",
    "points_at_close": "its story points when the sprint closed",
    "done_in_sprint": "true if it was finished in this sprint",
    "spilled_over": "at the story's first sprint: true if it was not done by the end (risk rule R1)",
    "reopened": "at the story's first sprint: true if it was reopened after done, then or in the next sprint (R6)",
    "started_at": "when work on the story started",
    "resolved_at": "when it was resolved",
    "hours_in_progress": "hours it spent in progress",
}
SPRINT_FIELDS = {"sprint_name": "name", "sprint_started_at": "started_at", "sprint_planned_end": "planned_end",
                 "sprint_closed_at": "closed_at"}
TIMES = ["sprint_started_at", "sprint_planned_end", "sprint_closed_at", "committed_at", "left_at", "started_at",
         "resolved_at"]
NUMBERS = ["points_at_commit", "points_at_close", "hours_in_progress"]
FLAGS = ["done_in_sprint", "spilled_over", "reopened"]
IN_A_SPRINT_ONLY = ["committed_at", "left_at", "points_at_commit", "points_at_close", *FLAGS]
TRUTH = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}


class StoredHistory:
    """Sprint records kept in this service's tables, in the record format (with each row's source)."""

    def records(self, project_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        with db.engine.connect() as connection:
            sprints = pd.read_sql(select(HistorySprint).where(HistorySprint.project_id == project_id), connection)
            items = pd.read_sql(select(HistoryItem).where(HistoryItem.project_id == project_id)
                                .order_by(HistoryItem.id), connection)
        return sprints, items


SOURCE = StoredHistory()
NO_TEAM = {"velocity_mean": None, "velocity_variance": None, "closed_sprints": None, "mean_cycle_time_hours": None,
           "spillover_rate": None, "reopen_rate": None}
NO_SPRINT = {"length_days": None, "parallel_sprints": None, "wip": None}


@dataclass(frozen=True)
class Copy:
    """A project's records as read at `read_at`: prepared (erp.serving.history.prepare), with their sources."""

    read_at: float
    sprints: pd.DataFrame
    items: pd.DataFrame
    sources: tuple[str, ...]

    @property
    def empty(self) -> bool:
        return self.sprints.empty and self.items.empty


_copies: dict[str, Copy] = {}
_refreshing: set[str] = set()
_lock = threading.Lock()


def _read(project_id: str) -> Copy | None:
    from erp.serving import history

    try:
        sprints, items = SOURCE.records(project_id)
    except SQLAlchemyError as error:
        store.note_failure(error)
        return None
    labels = set(sprints.get("source", [])) | set(items.get("source", []))
    copy = Copy(time.monotonic(), *history.prepare(sprints, items), tuple(sorted(labels)))
    _copies[project_id] = copy
    return copy


def _refresh(project_id: str) -> None:
    try:
        _read(project_id)
    finally:
        with _lock:
            _refreshing.discard(project_id)


def records(project_id: str) -> Copy | None:
    """The project's records: the worker's copy (re-read in the background once older than REFRESH_SECONDS), or
    read now for a project not seen yet; None while the database cannot be read."""
    copy = _copies.get(project_id)
    if copy is None:
        return None if store.recently_down() else _read(project_id)
    if time.monotonic() - copy.read_at > REFRESH_SECONDS and not store.recently_down():
        with _lock:
            start = project_id not in _refreshing
            _refreshing.add(project_id)
        if start:
            threading.Thread(target=_refresh, args=(project_id,), daemon=True).start()
    return copy


def context(project_id: str, sprint_id: str | None = None,
            exclude: frozenset[str] = frozenset()) -> tuple[dict, dict]:
    """The project's team and sprint context now, from its records (all None without records)."""
    from erp.serving import history

    copy = records(project_id)
    if copy is None or copy.empty:
        return dict(NO_TEAM), dict(NO_SPRINT)
    return history.context(copy.sprints, copy.items, project_id, pd.Timestamp.now(tz=UTC), sprint_id, exclude,
                           prepared=True)


def merged(given: dict | None, stored: dict, key: str) -> dict | None:
    """The caller's values where it gave them, the stored history's elsewhere. 'source' tells the engine who
    answered for the group: the caller when it gave `key`, the history otherwise."""
    given = {k: v for k, v in (given or {}).items() if v is not None}
    values = {**{k: v for k, v in stored.items() if v is not None}, **given}
    return {**values, "source": "request" if key in given else "history"} if values else None


def summary(project_id: str) -> dict:
    """What the models see about the project's team now, and its sprints, newest first."""
    from erp.models.confidence import COLD_START_SPRINTS
    from erp.serving import history

    copy = records(project_id)
    team, _ = context(project_id)
    sprints, items = (copy.sprints, copy.items) if copy else history.prepare(pd.DataFrame(), pd.DataFrame())
    done = items["points_at_close"].where(items["done_in_sprint"].fillna(False).astype(bool), 0).fillna(0)
    per_sprint = pd.DataFrame({
        "stories": items.groupby("sprint_id")["story_id"].nunique(),
        "committed_points": items.groupby("sprint_id")["points_at_commit"].sum(),
        "completed_points": done.groupby(items["sprint_id"]).sum(),
        "spilled_over": items["spilled_over"].fillna(False).astype(int).groupby(items["sprint_id"]).sum(),
    })
    listed = []
    for sprint in sprints.sort_values("started_at", ascending=False).itertuples():
        counts = per_sprint.loc[sprint.sprint_id] if sprint.sprint_id in per_sprint.index else None
        closed = pd.notna(sprint.closed_at)
        listed.append({
            "sprint_id": sprint.sprint_id, "name": sprint.name if pd.notna(sprint.name) else None,
            "started_at": sprint.started_at.tz_localize(UTC), "planned_end": sprint.planned_end.tz_localize(UTC),
            "closed_at": sprint.closed_at.tz_localize(UTC) if closed else None,
            "stories": int(counts["stories"]) if counts is not None else 0,
            "committed_points": float(counts["committed_points"]) if counts is not None else 0.0,
            "completed_points": float(counts["completed_points"]) if counts is not None and closed else None,
            "spilled_over": int(counts["spilled_over"]) if counts is not None and closed else None,
        })
    closed_sprints = team["closed_sprints"] or 0
    return {"project_id": project_id, "as_of": datetime.now(UTC), "sources": list(copy.sources) if copy else [],
            "closed_sprints": closed_sprints, "cold_start": closed_sprints < COLD_START_SPRINTS,
            "sprints_needed": max(0, COLD_START_SPRINTS - closed_sprints), "team_context": team, "sprints": listed}


# ------------------------------------------------------------------ writing records


def replace(project_id: str, sprints: pd.DataFrame, items: pd.DataFrame, source: str) -> dict:
    """Replace all of the project's records with these (a full history each time: importing again is safe)."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    with db.engine.begin() as connection:
        connection.execute(delete(HistoryItem).where(HistoryItem.project_id == project_id))
        connection.execute(delete(HistorySprint).where(HistorySprint.project_id == project_id))
        if len(sprints):
            connection.execute(insert(HistorySprint), _rows(sprints.assign(project_id=project_id, source=source)))
        for start in range(0, len(items), BATCH):
            connection.execute(insert(HistoryItem),
                               _rows(items.iloc[start:start + BATCH].assign(project_id=project_id, source=source)))
    _copies.pop(project_id, None)
    return {"sprints": len(sprints), "stories": int(items["story_id"].nunique()) if len(items) else 0,
            "rows": len(items)}


def remove(source: str | None = None, project_id: str | None = None) -> list[str]:
    """Delete records by source and/or project; the projects whose records went."""
    conditions_sprint = [HistorySprint.source == source] if source else []
    conditions_item = [HistoryItem.source == source] if source else []
    if project_id:
        conditions_sprint.append(HistorySprint.project_id == project_id)
        conditions_item.append(HistoryItem.project_id == project_id)
    with db.engine.begin() as connection:
        projects = set(connection.scalars(select(HistorySprint.project_id).where(*conditions_sprint)))
        projects |= set(connection.scalars(select(HistoryItem.project_id).where(*conditions_item)))
        connection.execute(delete(HistoryItem).where(*conditions_item))
        connection.execute(delete(HistorySprint).where(*conditions_sprint))
    for project in projects:
        _copies.pop(project, None)
    return sorted(projects)


def projects() -> dict[str, list[str]]:
    """Every project with records, and their sources."""
    with db.engine.connect() as connection:
        rows = connection.execute(select(HistorySprint.project_id, HistorySprint.source).distinct()).all()
        rows += connection.execute(select(HistoryItem.project_id, HistoryItem.source).distinct()).all()
    found: dict[str, set[str]] = {}
    for project, source in rows:
        found.setdefault(project, set()).add(source)
    return {project: sorted(sources) for project, sources in sorted(found.items())}


def _rows(frame: pd.DataFrame) -> list[dict]:
    """Rows for the database: times in UTC (a time without a zone is taken as UTC), None for every missing value,
    plain Python values otherwise."""
    frame = frame.copy()
    for column in ("started_at", "planned_end", "closed_at", "committed_at", "left_at", "resolved_at"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    frame = frame.astype(object)
    return frame.where(frame.notna(), None).to_dict("records")


def forget() -> None:
    """Drop the cached records (tests)."""
    _copies.clear()


# ------------------------------------------------------------------ sprint updates from the platform


def date_checks(values: dict[str, pd.Series]) -> list[tuple[pd.Series, str, str]]:
    """(where it is wrong, column, problem) for times out of order, one entry per story row: shared by the CSV
    import and the sprint updates."""
    start, end, closed = values["sprint_started_at"], values["sprint_planned_end"], values["sprint_closed_at"]
    return [
        (end <= start, "sprint_planned_end", "the sprint must end after it starts"),
        (closed < start, "sprint_closed_at", "the sprint cannot close before it starts"),
        (values["committed_at"] > closed, "committed_at", "committed after the sprint closed"),
        (values["left_at"] < values["committed_at"], "left_at", "left the sprint before it was committed"),
        (values["resolved_at"] < values["started_at"], "resolved_at", "resolved before work started"),
    ]


def sprint_frames(sprint_id: str, record: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """(sprint, its stories, problems) from a sprint record (app.schemas.SprintRecord), each problem named after
    the field it is in, e.g. stories[2].left_at."""
    from erp.serving.history import ITEM_COLUMNS

    sprint = pd.DataFrame([{"sprint_id": sprint_id, **{k: record.get(k) for k in SPRINT_FIELDS.values()}}])
    items = pd.DataFrame(record.get("stories") or [], columns=[c for c in ITEM_COLUMNS if c != "sprint_id"])
    items.insert(0, "sprint_id", sprint_id)
    rows = max(len(items), 1)  # the sprint's own dates are checked even without stories
    values = {f"sprint_{k}": pd.to_datetime(pd.Series([record.get(k)] * rows), utc=True)
              for k in ("started_at", "planned_end", "closed_at")}
    for column in ("committed_at", "left_at", "started_at", "resolved_at"):
        values[column] = pd.to_datetime(items[column] if len(items) else pd.Series([None]), utc=True)
    problems, seen = [], set()
    for wrong, column, text in date_checks(values):
        for row in wrong.index[wrong.fillna(False)]:
            field = column.removeprefix("sprint_") if column.startswith("sprint_") else f"stories[{row}].{column}"
            if field not in seen:
                seen.add(field)
                problems.append({"row": None, "column": field, "problem": text})
    for story, rows_of_story in items.groupby("story_id").groups.items():
        if len(rows_of_story) > 1:
            problems.append({"row": None, "column": f"stories[{rows_of_story[1]}].story_id",
                             "problem": f"{story} is already in this sprint as stories[{rows_of_story[0]}]"})
    return sprint, items, problems


def upsert_sprint(project_id: str, sprint: pd.DataFrame, items: pd.DataFrame, source: str = "platform") -> dict:
    """Replace one sprint of the project, and its stories, with this copy. At the close, a story sent without its
    R1 outcome gets it from done_in_sprint when this is its first sprint, and each story's prediction for this
    sprint gets its outcome (FR19)."""
    sprint_id, started = str(sprint.at[0, "sprint_id"]), pd.Timestamp(sprint.at[0, "started_at"])
    closed_at = sprint.at[0, "closed_at"]
    closed = closed_at is not None and not pd.isna(closed_at)
    items = items.copy()
    with db.engine.begin() as connection:
        if closed and len(items):
            earlier = set(connection.scalars(
                select(HistoryItem.story_id).join(HistorySprint, (HistoryItem.project_id == HistorySprint.project_id)
                                                  & (HistoryItem.sprint_id == HistorySprint.sprint_id))
                .where(HistoryItem.project_id == project_id, HistorySprint.sprint_id != sprint_id,
                       HistorySprint.started_at < started.to_pydatetime(),
                       HistoryItem.story_id.in_(items["story_id"].tolist()))))
            derive = items["spilled_over"].isna() & items["done_in_sprint"].notna() & ~items["story_id"].isin(earlier)
            items["spilled_over"] = items["spilled_over"].astype(object)
            items.loc[derive, "spilled_over"] = ~items.loc[derive, "done_in_sprint"].astype(bool)
        connection.execute(delete(HistoryItem).where(HistoryItem.project_id == project_id,
                                                     HistoryItem.sprint_id == sprint_id))
        connection.execute(delete(HistorySprint).where(HistorySprint.project_id == project_id,
                                                       HistorySprint.sprint_id == sprint_id))
        connection.execute(insert(HistorySprint), _rows(sprint.assign(project_id=project_id, source=source)))
        if len(items):
            connection.execute(insert(HistoryItem), _rows(items.assign(project_id=project_id, source=source)))
        outcomes = _sprint_outcomes(connection, project_id, sprint_id, items, closed_at) if closed else 0
    _copies.pop(project_id, None)
    return {"closed": closed, "stories": int(items["story_id"].nunique()) if len(items) else 0,
            "outcomes_recorded": outcomes}


def _sprint_outcomes(connection, project_id: str, sprint_id: str, items: pd.DataFrame, closed_at) -> int:
    """Each story's latest prediction for this sprint (made before it closed) gets the sprint's outcome; one
    already recorded for it is updated, since the platform's sprint is the record of what happened."""
    known = items[items["done_in_sprint"].notna()].drop_duplicates("story_id", keep="last").set_index("story_id")
    if known.empty:
        return 0
    predictions = connection.execute(
        select(PredictionRecord.id, PredictionRecord.story_id).where(
            PredictionRecord.project_id == project_id, PredictionRecord.sprint_id == sprint_id,
            PredictionRecord.story_id.in_(known.index.tolist()),
            PredictionRecord.created_at <= pd.Timestamp(closed_at).to_pydatetime())
        .order_by(PredictionRecord.created_at)).all()
    latest = {story: prediction for prediction, story in predictions}  # the last one per story
    if not latest:
        return 0
    existing = dict(connection.execute(
        select(OutcomeRecord.prediction_id, OutcomeRecord.id).where(
            OutcomeRecord.prediction_id.in_(list(latest.values()))).order_by(OutcomeRecord.created_at)).all())
    for story, prediction in latest.items():
        row = known.loc[story]
        values = {"completed_in_sprint": bool(row["done_in_sprint"]),
                  "actual_story_points": None if pd.isna(row["points_at_close"]) else float(row["points_at_close"]),
                  "reopened": bool(row["reopened"]) if not pd.isna(row["reopened"]) else False}
        if prediction in existing:
            connection.execute(update(OutcomeRecord).where(OutcomeRecord.id == existing[prediction]).values(**values))
        else:
            connection.execute(insert(OutcomeRecord).values(id=str(uuid.uuid4()), prediction_id=prediction,
                                                            created_at=datetime.now(UTC), **values))
    return len(latest)


def remove_sprint(project_id: str, sprint_id: str) -> bool:
    """Delete one sprint of the project and its stories (outcomes already recorded stay)."""
    with db.engine.begin() as connection:
        connection.execute(delete(HistoryItem).where(HistoryItem.project_id == project_id,
                                                     HistoryItem.sprint_id == sprint_id))
        gone = connection.execute(delete(HistorySprint).where(HistorySprint.project_id == project_id,
                                                              HistorySprint.sprint_id == sprint_id)).rowcount
    _copies.pop(project_id, None)
    return bool(gone)


# ------------------------------------------------------------------ the CSV import


def to_csv(sprints: pd.DataFrame, items: pd.DataFrame) -> str:
    """Records (the record format) as the import's CSV, in the items' order."""
    table = items.reset_index(drop=True).merge(
        sprints.rename(columns={v: k for k, v in SPRINT_FIELDS.items()}).drop(columns="project_id", errors="ignore"),
        on="sprint_id", how="left", sort=False).reindex(columns=list(CSV_COLUMNS))
    for column in TIMES:
        times = pd.to_datetime(table[column], utc=True)
        table[column] = times.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").where(times.notna(), "")
    for column in FLAGS:
        table[column] = table[column].map({True: "true", False: "false"}).fillna("")
    return table.to_csv(index=False, na_rep="")


def parse_csv(text: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """(sprints, items, problems): the records in a CSV, or what is wrong with it (row numbers as a spreadsheet
    shows them, the header being row 1). Nothing is imported while there are problems."""
    problems: list[dict] = []

    def problem(row: int | None, column: str | None, text: str) -> None:
        problems.append({"row": row, "column": column, "problem": text})

    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        problem(None, None, f"not a CSV file: {error}")
        return pd.DataFrame(), pd.DataFrame(), problems
    frame.columns = [str(c).strip() for c in frame.columns]
    for column in sorted(set(frame.columns) - set(CSV_COLUMNS)):
        problem(None, column, "unknown column (see the README for the columns)")
    if "story_id" not in frame.columns:
        problem(None, "story_id", "the story_id column is required")
        return pd.DataFrame(), pd.DataFrame(), problems
    if len(frame) > MAX_ROWS:
        problem(None, None, f"more than {MAX_ROWS:,} rows")
        return pd.DataFrame(), pd.DataFrame(), problems

    frame = frame.reindex(columns=list(CSV_COLUMNS), fill_value="").apply(lambda column: column.str.strip())
    frame.index = frame.index + 2
    given = frame != ""
    values: dict[str, pd.Series] = {}
    for column in TIMES:
        values[column] = pd.to_datetime(frame[column].where(given[column]), utc=True, errors="coerce",
                                        format="ISO8601")
        for row in frame.index[given[column] & values[column].isna()]:
            problem(int(row), column, "not a date and time; use ISO 8601, e.g. 2026-09-14T09:00:00Z")
    for column in NUMBERS:
        values[column] = pd.to_numeric(frame[column].where(given[column]), errors="coerce")
        for row in frame.index[given[column] & (values[column].isna() | (values[column] < 0))]:
            problem(int(row), column, "not a number of zero or more")
    for column in FLAGS:
        lowered = frame[column].str.lower()
        values[column] = lowered.map(TRUTH).astype("boolean")
        for row in frame.index[given[column] & ~lowered.isin(TRUTH)]:
            problem(int(row), column, "not true or false")

    in_sprint = given["sprint_id"]
    for row in frame.index[~given["story_id"]]:
        problem(int(row), "story_id", "every row needs a story_id")
    for column in ("sprint_started_at", "sprint_planned_end", "committed_at"):
        for row in frame.index[in_sprint & ~given[column]]:
            problem(int(row), column, "required for a story in a sprint")
    for column in [*SPRINT_FIELDS, *IN_A_SPRINT_ONLY]:
        for row in frame.index[~in_sprint & given[column]]:
            problem(int(row), column, "only for a story in a sprint (this row has no sprint_id)")
    for rows in frame[in_sprint].groupby(["sprint_id", "story_id"]).groups.values():
        if len(rows) > 1:
            problem(int(rows[1]), "story_id", f"the story is already in this sprint on row {rows[0]}")
    in_a_sprint = set(frame.loc[in_sprint, "story_id"])
    for row in frame.index[~in_sprint & frame["story_id"].isin(in_a_sprint)]:
        problem(int(row), "sprint_id", "this story is in a sprint on another row, so it cannot be outside one")
    for rows in frame[~in_sprint].groupby("story_id").groups.values():
        if len(rows) > 1:
            problem(int(rows[1]), "story_id", f"the same story outside a sprint as on row {rows[0]}")

    for sprint_id, rows in frame[in_sprint].groupby("sprint_id").groups.items():
        for column in SPRINT_FIELDS:
            seen = values[column][rows] if column in values else frame.loc[rows, column]
            if seen.nunique(dropna=False) > 1:
                problem(int(rows[0]), column, f"sprint {sprint_id} has different values on rows "
                                              f"{', '.join(str(r) for r in rows)}")
    for wrong, column, text in date_checks(values):
        for row in frame.index[wrong.fillna(False)]:
            problem(int(row), column, text)
    if problems:
        return pd.DataFrame(), pd.DataFrame(), sorted(problems, key=lambda p: (p["row"] or 0, p["column"] or ""))

    table = frame.assign(**values)
    sprints = table[in_sprint].drop_duplicates("sprint_id")[["sprint_id", *SPRINT_FIELDS]].rename(
        columns=SPRINT_FIELDS)
    sprints["name"] = sprints["name"].where(sprints["name"] != "")
    items = table[["sprint_id", "story_id", "issue_type", *IN_A_SPRINT_ONLY[:4], *FLAGS, "started_at",
                   "resolved_at", "hours_in_progress"]].copy()
    items["sprint_id"] = items["sprint_id"].where(in_sprint)
    items["issue_type"] = items["issue_type"].where(items["issue_type"] != "")
    return sprints.reset_index(drop=True), items.reset_index(drop=True), []
