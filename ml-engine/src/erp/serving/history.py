"""Team history from sprint records (FR5), computed with the training pipeline's own code (erp.features.team).

The platform owns the sprints. Until it offers them through an API, the service keeps copies of sprint records
(backend/app/history.py); either way they arrive in this shape (the record format, README "Sprint history"):

  sprints  one row per sprint: project_id, sprint_id, name, started_at, planned_end, closed_at (empty while the
           sprint runs)
  items    one row per story committed to a sprint (a story carried into the next sprint has a row in each):
           project_id, sprint_id, story_id, issue_type, committed_at, left_at (empty: still in the sprint at the
           end), points_at_commit, points_at_close, done_in_sprint, spilled_over and reopened (the story's outcome
           at its first commitment, where the platform tracked it: risk rules R1 and R6), started_at, resolved_at,
           hours_in_progress. A story never in a sprint has one row with an empty sprint_id: it counts only
           towards the cycle time, which training averaged over all of a project's resolved stories.

features() gives each (project, moment) the team-history and sprint-timing features exactly as
erp.features.build_features computed them for training, from what was known at that moment only:
  - a sprint counts once it had closed; its velocity is the points (at the close) of the stories done in it;
  - a spillover is known 12 h after its sprint closed (the TAWOS clock tolerance), a reopening one sprint length
    after it; each rate uses the project's last 50 known outcomes and needs at least 5;
  - cycle time: the last 50 resolved stories of the story types, known when they were resolved.
All times are UTC.
"""

import numpy as np
import pandas as pd

from erp.features import team
from erp.snapshot.filters import STORY_TYPES
from erp.snapshot.timeline import CLOCK_TOLERANCE

SPRINT_COLUMNS = ["project_id", "sprint_id", "name", "started_at", "planned_end", "closed_at"]
ITEM_COLUMNS = ["project_id", "sprint_id", "story_id", "issue_type", "committed_at", "left_at", "points_at_commit",
                "points_at_close", "done_in_sprint", "spilled_over", "reopened", "started_at", "resolved_at",
                "hours_in_progress"]
SPRINT_TIMES = ["started_at", "planned_end", "closed_at"]
ITEM_TIMES = ["committed_at", "left_at", "started_at", "resolved_at"]
FLAGS = ["done_in_sprint", "spilled_over", "reopened"]
TEAM = ["team_velocity_rolling", "velocity_variance", "history_sprints", "mean_cycle_time_hours",
        "historical_spillover_rate", "reopen_rate"]
SPRINT = ["sprint_length_days", "parallel_sprints", "wip_at_commitment"]
DAY = pd.Timedelta(days=1)


def _utc(values: pd.Series) -> pd.Series:
    """Naive UTC timestamps (the training data's times are naive)."""
    values = pd.to_datetime(values, utc=True)
    return values.dt.tz_convert(None).astype("datetime64[us]")


def prepare(sprints: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The records with their types settled: UTC times, numbers, and flags that may be unknown."""
    sprints = sprints.reindex(columns=SPRINT_COLUMNS).copy()
    items = items.reindex(columns=ITEM_COLUMNS).copy()
    for frame, times in ((sprints, SPRINT_TIMES), (items, ITEM_TIMES)):
        for column in times:
            frame[column] = _utc(frame[column])
        for column in ("project_id", "sprint_id"):
            frame[column] = frame[column].where(frame[column].notna(), "").astype(str)
    items["story_id"] = items["story_id"].astype(str)
    for column in ("points_at_commit", "points_at_close", "hours_in_progress"):
        items[column] = pd.to_numeric(items[column], errors="coerce").astype(float)
    for column in FLAGS:
        items[column] = items[column].astype("boolean")
    return sprints, items


def features(sprints: pd.DataFrame, items: pd.DataFrame, stories: pd.DataFrame,
             exclude: frozenset[str] = frozenset(), prepared: bool = False) -> pd.DataFrame:
    """The team-history and sprint-timing features of each story.

    stories: Project_ID, snapshot_time (the moment) and Sprint_ID (the story's own sprint, possibly not in the
    records yet, e.g. one being planned). exclude: story ids left out of the work in progress (the stories being
    predicted are not 'other' stories of their sprint). prepared: the records already went through prepare().
    """
    if not prepared:
        sprints, items = prepare(sprints, items)
    stories = stories.assign(Project_ID=stories["Project_ID"].astype(str),
                             snapshot_time=_utc(stories["snapshot_time"]))
    keys = ["project_id", "sprint_id"]
    of_sprint = sprints.set_index(keys)[["started_at", "planned_end", "closed_at"]].add_prefix("sprint_")
    items = items.join(of_sprint, on=keys)

    # Velocity: points (at the close) of the stories done in each closed sprint, as training's sprint_outcomes.
    closed = sprints[sprints["closed_at"].notna()]
    done = items["points_at_close"].where(items["done_in_sprint"].fillna(False).astype(bool), 0).fillna(0)
    finished = done.groupby([items["project_id"], items["sprint_id"]]).sum()
    velocity = pd.DataFrame({
        "Project_ID": closed["project_id"].to_numpy(), "closed_at": closed["closed_at"].to_numpy(),
        "completed_points": finished.reindex(pd.MultiIndex.from_frame(closed[keys]), fill_value=0).to_numpy()})
    out = team.previous_sprint_stats(velocity, stories).astype(float)

    # Outcome rates, each outcome counting once it could be known.
    spilled = items[items["spilled_over"].notna() & items["sprint_closed_at"].notna()]
    out["historical_spillover_rate"] = team.recent_rate(pd.DataFrame({
        "Project_ID": spilled["project_id"], "r1": spilled["spilled_over"].astype(float),
        "r1_known": spilled["sprint_closed_at"] + CLOCK_TOLERANCE}), stories, "r1", "r1_known")
    reopened = items[items["reopened"].notna() & items["sprint_closed_at"].notna()]
    out["reopen_rate"] = team.recent_rate(pd.DataFrame({
        "Project_ID": reopened["project_id"], "r6": reopened["reopened"].astype(float),
        "r6_known": reopened["sprint_closed_at"] + (reopened["sprint_closed_at"] - reopened["sprint_started_at"])}),
        stories, "r6", "r6_known")

    # Cycle time: each story once, when resolved, of the story types (a story without a type counts).
    story = items.drop_duplicates(["project_id", "story_id"], keep="last")
    story = story[story["hours_in_progress"].notna() & story["resolved_at"].notna()
                  & (story["issue_type"].isna() | story["issue_type"].isin(STORY_TYPES))]
    out["mean_cycle_time_hours"] = team.recent_rate(pd.DataFrame({
        "Project_ID": story["project_id"], "hours": story["hours_in_progress"], "known": story["resolved_at"]}),
        stories, "hours", "known")

    sprint = _sprint_timing(sprints, items, stories, exclude)
    return pd.concat([out[TEAM], sprint], axis=1)


def _sprint_timing(sprints: pd.DataFrame, items: pd.DataFrame, stories: pd.DataFrame,
                   exclude: frozenset[str]) -> pd.DataFrame:
    """Length of the story's own sprint, the project's other sprints running, and the own sprint's other stories
    already in progress, at the moment."""
    out = pd.DataFrame(np.nan, index=stories.index, columns=SPRINT)
    for project, group in stories.groupby("Project_ID"):
        own = sprints[sprints["project_id"] == project]
        if own.empty:
            continue
        t = group["snapshot_time"].to_numpy(dtype="datetime64[us]")[:, None]
        starts = own["started_at"].to_numpy(dtype="datetime64[us]")[None, :]
        closes = own["closed_at"].to_numpy(dtype="datetime64[us]")[None, :]
        running = (starts <= t).sum(axis=1) - (closes <= t).sum(axis=1)  # a sprint still open never closed
        known = group["Sprint_ID"].astype(str).isin(own["sprint_id"]).to_numpy()
        # Training counted the story's own sprint out (it runs at its snapshot); a sprint being planned is not
        # in the records yet, so there is nothing to count out.
        out.loc[group.index, "parallel_sprints"] = np.maximum(running - known, 0)
        lengths = own.set_index("sprint_id")
        length = (lengths["planned_end"] - lengths["started_at"]) / DAY
        out.loc[group.index, "sprint_length_days"] = group["Sprint_ID"].astype(str).map(length).to_numpy()
        project_items = items[(items["project_id"] == project) & ~items["story_id"].isin(exclude)]
        by_sprint = dict(tuple(project_items.groupby("sprint_id")))
        for index, sprint_id, moment in zip(group.index, group["Sprint_ID"].astype(str), group["snapshot_time"],
                                            strict=True):
            if sprint_id not in lengths.index:
                continue
            mine = by_sprint.get(sprint_id, project_items.iloc[:0])
            present = (mine["committed_at"] <= moment) & (mine["left_at"].isna() | (mine["left_at"] > moment))
            started = (mine["started_at"] <= moment) & (mine["resolved_at"].isna() | (mine["resolved_at"] > moment))
            out.at[index, "wip_at_commitment"] = float((present & started).sum())
    return out


def context(sprints: pd.DataFrame, items: pd.DataFrame, project_id: str, at: pd.Timestamp,
            sprint_id: str | None = None, exclude: frozenset[str] = frozenset(),
            prepared: bool = False) -> tuple[dict, dict]:
    """A project's team and sprint context at a moment, in the API's field names (TeamContext, SprintContext);
    unknown values are None."""
    stories = pd.DataFrame({"Project_ID": [str(project_id)], "snapshot_time": [at],
                            "Sprint_ID": [str(sprint_id) if sprint_id is not None else ""]})
    row = features(sprints, items, stories, exclude, prepared).iloc[0]

    def value(name: str) -> float | None:
        return None if pd.isna(row[name]) else float(row[name])

    closed = value("history_sprints")
    team_context = {"velocity_mean": value("team_velocity_rolling"), "velocity_variance": value("velocity_variance"),
                    "closed_sprints": int(closed) if closed is not None else None,
                    "mean_cycle_time_hours": value("mean_cycle_time_hours"),
                    "spillover_rate": value("historical_spillover_rate"), "reopen_rate": value("reopen_rate")}
    parallel, wip = value("parallel_sprints"), value("wip_at_commitment")
    sprint_context = {"length_days": value("sprint_length_days"),
                      "parallel_sprints": int(parallel) if parallel is not None else None,
                      "wip": int(wip) if wip is not None else None}
    return team_context, sprint_context
