"""Team clues: how the project's earlier sprints went, and what the sprint looked like at commitment.

"Team" means the project: TAWOS has no board or team field. In projects that run several sprints at once
(several teams), the previous sprints are the project's, whichever team ran them; parallel_sprints tells
the model how many were running.

Every clue uses only what was known at the story's snapshot time: a previous sprint counts once it had
closed, a past story's outcome once its label could be known (its sprint's close for R1, one more sprint
length for R6), and other issues' status and story points as they were at that moment.
"""

import numpy as np
import pandas as pd

# Statuses that mean "not started" in the TAWOS workflows; any other status of an unresolved issue means
# work has started (In Progress, In Review, Waiting for testing, ...).
NOT_STARTED = {
    "open", "to do", "todo", "new", "backlog", "reopened", "accepted", "needs triage", "needs scheduling",
    "gathering interest", "gathering impact", "long term backlog", "awaiting development",
    "not being considered", "selected for development", "ready for development", "ready", "to be reviewed",
    "needs verification", "triage", "planned", "defined",
}
VELOCITY_SPRINTS = 3  # team_velocity_rolling: mean over the last 3 closed sprints (ML guide Appendix B.1)
VARIANCE_SPRINTS = 5  # velocity_variance: over the last 5
HISTORY_STORIES = 50  # spillover / reopen / cycle time: the project's last 50 stories with a known outcome
MIN_HISTORY = 5       # fewer than this and the rate is left missing (cold start)


def is_started(status: pd.Series, resolved: pd.Series) -> pd.Series:
    """Work had started: unresolved, and the status is not one of the 'not started' ones."""
    return ~resolved & status.notna() & ~status.fillna("").str.strip().str.lower().isin(NOT_STARTED)


def previous_sprint_stats(sprints: pd.DataFrame, stories: pd.DataFrame) -> pd.DataFrame:
    """Velocity clues per story from the project's sprints that had closed by the story's snapshot time.

    sprints: one row per (Project_ID, Sprint_ID) with closed_at and completed_points.
    stories: index Issue_ID, with Project_ID and snapshot_time.
    """
    out = pd.DataFrame(index=stories.index, columns=["team_velocity_rolling", "velocity_variance", "history_sprints"],
                       dtype=float)
    for project, group in stories.groupby("Project_ID"):
        done = sprints[sprints["Project_ID"] == project].sort_values("closed_at")
        closed = done["closed_at"].to_numpy(dtype="datetime64[us]")
        points = done["completed_points"].to_numpy(dtype=float)
        counts = np.searchsorted(closed, group["snapshot_time"].to_numpy(dtype="datetime64[us]"), side="right")
        for issue_id, n in zip(group.index, counts, strict=True):
            last = points[max(0, n - VELOCITY_SPRINTS):n]
            spread = points[max(0, n - VARIANCE_SPRINTS):n]
            out.loc[issue_id] = [last.mean() if len(last) else np.nan,
                                 spread.var(ddof=1) if len(spread) >= 2 else np.nan, n]
    return out


def recent_rate(events: pd.DataFrame, stories: pd.DataFrame, value: str, known: str,
                window: int = HISTORY_STORIES, minimum: int = MIN_HISTORY) -> pd.Series:
    """Mean of `value` over the project's last `window` events known by each story's snapshot time.

    events: Project_ID, the value column and the time it became known. Leaves NaN below `minimum` events.
    """
    out = pd.Series(np.nan, index=stories.index)
    for project, group in stories.groupby("Project_ID"):
        past = events[events["Project_ID"] == project].sort_values(known)
        times = past[known].to_numpy(dtype="datetime64[us]")
        cumulative = np.concatenate([[0.0], np.cumsum(past[value].to_numpy(dtype=float))])
        counts = np.searchsorted(times, group["snapshot_time"].to_numpy(dtype="datetime64[us]"), side="right")
        for issue_id, n in zip(group.index, counts, strict=True):
            k = min(n, window)
            if k >= minimum:
                out[issue_id] = (cumulative[n] - cumulative[n - k]) / k
    return out
