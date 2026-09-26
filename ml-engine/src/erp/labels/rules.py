"""The building blocks of the R1-R6 rules: events after a moment, blocking links, and time spent blocked.

Everything here works on plain tables of events (issue, time) so each piece can be tested on its own.
How the pieces combine into the six rules is in build_labels.py.
"""

import re

import numpy as np
import pandas as pd

from erp.snapshot import history

# R4: blocked for more than this share of the sprint (ML guide 6.3: "more than 30% of the sprint").
BLOCKED_SHARE = 0.30
# Link phrases, as the Change_Log writes them, that mean "another issue has to be finished first".
DEPENDS_ON = ("is blocked by", "depends on", "has to be done after")
BLOCKING_LINK = re.compile(r"^This issue (?:" + "|".join(DEPENDS_ON) + r") ([A-Z][A-Z0-9_]*-\d+)$", re.I)


def first_event_after(events: pd.DataFrame, after: pd.Series) -> pd.Series:
    """For each issue in `after` (index Issue_ID), the time of its first event strictly after that moment."""
    rows = events.merge(after.rename("_after"), left_on="Issue_ID", right_index=True)
    rows = rows[rows["Creation_Date"] > rows["_after"]]
    return rows.groupby("Issue_ID")["Creation_Date"].min().reindex(after.index)


def blocking_links(link_changes: pd.DataFrame, issue_ids: pd.Series) -> pd.DataFrame:
    """When each dependency link ('is blocked by', 'depends on', 'has to be done after') existed.

    One row per link and period: Issue_ID, blocker_key, added_at, removed_at (NaT while it exists) and
    Blocker_ID (NA for blockers outside TAWOS). See history.link_periods.
    """
    links = history.link_periods(link_changes, issue_ids)
    links = links[links["phrase"].isin(DEPENDS_ON)].reset_index(drop=True)
    return links.rename(columns={"target_key": "blocker_key", "Target_ID": "Blocker_ID"})[
        ["Issue_ID", "blocker_key", "added_at", "removed_at", "Blocker_ID"]]


def open_periods(resolution_changes: pd.DataFrame, created: pd.Series) -> pd.DataFrame:
    """When each issue was unresolved: one row per period (opened_at, closed_at; NaT while still open).

    resolution_changes holds Change_Log rows for Field == 'resolution' (ID, Issue_ID, To_String,
    Creation_Date): a value marks the issue resolved, an empty value reopens it. An issue is open from
    its creation until its first resolution.
    """
    rows = []
    log = resolution_changes.sort_values(["Issue_ID", "Creation_Date", "ID"])
    by_issue = dict(tuple(log.groupby("Issue_ID", sort=False)))
    for issue_id, since in created.items():
        opened = since
        changes = by_issue.get(issue_id)
        if changes is not None:
            for when, value in zip(changes["Creation_Date"], changes["To_String"], strict=True):
                resolved = isinstance(value, str) and value.strip() != ""
                if resolved and opened is not None:
                    rows.append((issue_id, opened, when))
                    opened = None
                elif not resolved and opened is None:
                    opened = when
        if opened is not None:
            rows.append((issue_id, opened, pd.NaT))
    periods = pd.DataFrame(rows, columns=["Issue_ID", "opened_at", "closed_at"])
    periods["closed_at"] = pd.to_datetime(periods["closed_at"])
    return periods


def covered_time(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Timedelta:
    """Total time covered by a set of (start, end) intervals, counting overlaps once."""
    total, current_start, current_end = pd.Timedelta(0), None, None
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def blocked_time(links: pd.DataFrame, blocker_open: pd.DataFrame, windows: pd.DataFrame) -> pd.Series:
    """How long each story was blocked inside its window: a blocking link existed and the blocker was open.

    windows: index Issue_ID, columns start and end (the story's time in the sprint). Blockers outside
    TAWOS are ignored because their state is unknown. Returns a Timedelta per story (0 when never blocked).
    """
    result = pd.Series(pd.Timedelta(0), index=windows.index)
    known = links[links["Blocker_ID"].notna() & links["Issue_ID"].isin(windows.index)]
    periods = blocker_open.groupby("Issue_ID")
    far_future = pd.Timestamp.max
    for issue_id, story_links in known.groupby("Issue_ID"):
        start, end = windows.at[issue_id, "start"], windows.at[issue_id, "end"]
        pieces = []
        for link in story_links.itertuples():
            link_end = link.removed_at if pd.notna(link.removed_at) else far_future
            if int(link.Blocker_ID) not in periods.groups:
                continue
            for period in periods.get_group(int(link.Blocker_ID)).itertuples():
                closed = period.closed_at if pd.notna(period.closed_at) else far_future
                lo = max(start, link.added_at, period.opened_at)
                hi = min(end, link_end, closed)
                if hi > lo:
                    pieces.append((lo, hi))
        result[issue_id] = covered_time(pieces)
    return pd.to_timedelta(result)


def risk_level(n_rules: pd.Series) -> pd.Series:
    """Proposal Appendix B: low when no rule fires, medium for one, high for two or more."""
    return pd.Series(np.select([n_rules == 0, n_rules == 1], ["low", "medium"], default="high"),
                     index=n_rules.index)
