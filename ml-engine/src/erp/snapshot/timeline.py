"""Rebuild which sprints each issue was in, and when, from the Change_Log.

TAWOS keeps only one sprint per issue (Issue.Sprint_ID, the sprint it was delivered in), so a story
that spilled over from Sprint 57 into Sprint 58 looks as if it was only ever in 58. The Change_Log
records every edit to the issue's Sprint field, and each edit stores the whole list of sprints after
the change, so the membership history can be replayed. Two Jira behaviours appear and both work:

- keep: a closed sprint stays in the list when the issue moves on ("103" -> "103, 106");
- replace: the old sprint is removed as the issue moves ("107" -> "108").

Sprint-table dates and Change_Log timestamps are not always on the same clock. Copies of one
sprint differ by exactly an hour between projects, and the bulk moves made when a sprint is closed
sit up to about nine hours away from Sprint.Complete_Date in some projects (see
reports/sprint-timeline.md). Sprint boundaries are therefore compared with CLOCK_TOLERANCE.
"""

import numpy as np
import pandas as pd

from erp import tawos

CLOCK_TOLERANCE = pd.Timedelta(hours=12)
# A stay shorter than this that ends while the sprint is still running is a correction, not a commitment
# (e.g. added to the wrong sprint and moved out again 20 minutes later).
MIN_STAY = pd.Timedelta(days=1)

STAY_COLUMNS = ["Issue_ID", "JiraID", "joined_at", "left_at", "join_logged"]
SPRINT_COLUMNS = ["Sprint_ID", "Sprint_Name", "State", "Start_Date", "End_Date", "Complete_Date"]


def replay_sprint_field(sprint_log: pd.DataFrame, created: pd.Series) -> pd.DataFrame:
    """One row per stay of an issue in a sprint: when the sprint entered its Sprint field and when it left.

    sprint_log holds the Change_Log rows for Field == 'Sprint' (ID, Issue_ID, From_Value, To_Value,
    Creation_Date); created maps Issue_ID to the issue's creation date. Sprints already present in the
    first change's From_Value were set when the issue was created or by an edit the log does not show,
    so they get the creation date and join_logged = False. left_at is NaT while the sprint is still in
    the field at the end of the log.
    """
    rows = []
    log = sprint_log.sort_values(["Issue_ID", "Creation_Date", "ID"])
    for issue_id, changes in log.groupby("Issue_ID", sort=False):
        current: dict[int, tuple[pd.Timestamp, bool]] = {}
        for sprint in tawos.parse_sprint_ids(changes["From_Value"].iloc[0]):
            current[sprint] = (created.get(issue_id, pd.NaT), False)
        for when, value in zip(changes["Creation_Date"], changes["To_Value"], strict=True):
            after = tawos.parse_sprint_ids(value)
            for sprint in [s for s in current if s not in after]:
                joined, logged = current.pop(sprint)
                rows.append((issue_id, sprint, joined, when, logged))
            for sprint in after:
                if sprint not in current:
                    current[sprint] = (when, True)
        rows += [(issue_id, sprint, joined, pd.NaT, logged) for sprint, (joined, logged) in current.items()]
    stays = pd.DataFrame(rows, columns=STAY_COLUMNS)
    stays["left_at"] = pd.to_datetime(stays["left_at"])
    return stays


def resolve_sprints(stays: pd.DataFrame, issues: pd.DataFrame, sprints: pd.DataFrame,
                    projects: pd.DataFrame) -> pd.DataFrame:
    """Attach the TAWOS Sprint row (and its dates) that each stay's Jira sprint ID refers to.

    Jira sprint IDs are unique only within one Jira instance (a TAWOS repository), and a sprint shared
    by several projects is stored once per project. Prefer the copy in the issue's own project, then
    the lowest-ID copy in the same repository. IDs that match neither stay unresolved (Sprint_ID NA).
    A Sprint_ID already on the stay (from Issue.Sprint_ID) is kept.
    """
    repo_of_project = projects.set_index("ID")["Repository_ID"]
    out = stays.merge(issues[["ID", "Project_ID"]].rename(columns={"ID": "Issue_ID"}), on="Issue_ID", how="left")
    out["Repository_ID"] = out["Project_ID"].map(repo_of_project)

    by_project = sprints.sort_values("ID").drop_duplicates(["Project_ID", "JiraID"])
    by_project = by_project.set_index(["Project_ID", "JiraID"])["ID"].rename("own")
    in_repo = sprints.assign(Repository_ID=sprints["Project_ID"].map(repo_of_project))
    in_repo = in_repo.sort_values("ID").drop_duplicates(["Repository_ID", "JiraID"])
    in_repo = in_repo.set_index(["Repository_ID", "JiraID"])["ID"].rename("repo")
    out = out.join(by_project, on=["Project_ID", "JiraID"]).join(in_repo, on=["Repository_ID", "JiraID"])

    known = out["Sprint_ID"] if "Sprint_ID" in out else pd.Series(pd.NA, index=out.index)
    out["Sprint_ID"] = known.fillna(out["own"]).fillna(out["repo"]).astype("Int64")
    details = sprints.set_index("ID").rename(columns={"Name": "Sprint_Name"})
    details = details[[c for c in SPRINT_COLUMNS if c != "Sprint_ID"]]
    return out.drop(columns=["own", "repo"]).join(details, on="Sprint_ID")


def sprint_memberships(stays: pd.DataFrame, tolerance: pd.Timedelta = CLOCK_TOLERANCE) -> pd.DataFrame:
    """One row per (issue, sprint): was the issue committed to the sprint, how it got there, how it left.

    entry says how the issue relates to the sprint (only the first two are commitments):
      planned            in the sprint at its start (start + tolerance), i.e. picked in sprint planning
      added_mid_sprint   joined after the start and at least the tolerance before the sprint was closed
      brief_stay         in and out again within MIN_STAY while the sprint was running (a correction)
      joined_at_close    joined within the tolerance of the close, i.e. while the sprint was being closed
      joined_after_close the sprint was already closed when it was added (a history edit)
      left_before_start  was in the sprint's field but gone before it started (re-planned)
      sprint_not_started the sprint has no start date (a future sprint)
      sprint_bad_dates   the sprint closed before it started (3 sprints in TAWOS)
      sprint_unknown     the Jira sprint ID is not in the Sprint table, so there are no dates
    exit (committed memberships only) says how the issue left the sprint:
      stayed             still in the sprint's field at the end of the log
      left_at_close      removed within the tolerance of the sprint's close (moved on when it closed)
      left_mid_sprint    removed while the sprint was running
      left_after_close   removed later, from an already closed sprint
      sprint_open        the sprint had not been closed when TAWOS was collected
    commit_order numbers the committed sprints of each issue by commitment time (1 = first commitment).
    """
    s = stays.copy()
    joined, left = s["joined_at"], s["left_at"]
    start_ref = s["Start_Date"] + tolerance
    close = s["Complete_Date"]
    close_ref = close - tolerance
    still_in = left.isna()
    in_running_sprint = left - joined.where(joined > s["Start_Date"], s["Start_Date"])
    s["_open"] = still_in
    s["_brief"] = (left > start_ref) & (close.isna() | (left < close_ref)) & (in_running_sprint < MIN_STAY)
    s["_in_at_start"] = (joined <= start_ref) & (still_in | (left > start_ref)) & ~s["_brief"]
    s["_joined_mid"] = (joined > start_ref) & (close.isna() | (joined < close_ref)) & ~s["_brief"]
    s["_at_close"] = (joined >= close_ref) & (joined <= close + tolerance)
    s["_in_at_close"] = (joined <= close) & (still_in | (left >= close_ref))
    s["_mid_join"] = joined.where(s["_joined_mid"])

    firsts = {c: (c, "first") for c in ["Project_ID", "Repository_ID", *SPRINT_COLUMNS]}
    m = s.groupby(["Issue_ID", "JiraID"], sort=False).agg(
        **firsts,
        first_joined=("joined_at", "min"),
        last_left=("left_at", "max"),
        still_in=("_open", "any"),
        in_at_start=("_in_at_start", "any"),
        joined_mid_sprint=("_joined_mid", "any"),
        brief=("_brief", "any"),
        joined_at_close=("_at_close", "any"),
        in_at_close=("_in_at_close", "any"),
        first_mid_join=("_mid_join", "min"),
        stays=("joined_at", "size"),
    ).reset_index()
    m["Sprint_ID"] = m["Sprint_ID"].astype("Int64")
    m["last_left"] = m["last_left"].where(~m["still_in"])

    bad_dates = m["Complete_Date"] < m["Start_Date"]
    usable = m["Sprint_ID"].notna() & m["Start_Date"].notna() & ~bad_dates
    m["committed"] = usable & (m["in_at_start"] | m["joined_mid_sprint"])
    m["entry"] = np.select(
        [m["Sprint_ID"].isna(), m["Start_Date"].isna(), bad_dates, m["in_at_start"], m["joined_mid_sprint"],
         m["joined_at_close"], m["first_joined"] > m["Complete_Date"] + tolerance, m["brief"]],
        ["sprint_unknown", "sprint_not_started", "sprint_bad_dates", "planned", "added_mid_sprint",
         "joined_at_close", "joined_after_close", "brief_stay"],
        default="left_before_start",
    )
    m["commitment_time"] = m["Start_Date"].where(m["in_at_start"], m["first_mid_join"]).where(m["committed"])
    close = m["Complete_Date"]
    m["exit"] = np.select(
        [~m["committed"], close.isna(), m["still_in"], m["last_left"] < close - tolerance,
         m["last_left"] <= close + tolerance],
        ["", "sprint_open", "stayed", "left_mid_sprint", "left_at_close"],
        default="left_after_close",
    )
    m["exit"] = m["exit"].where(m["committed"])

    order = m[m["committed"]].sort_values(["Issue_ID", "commitment_time", "Start_Date", "Sprint_ID"])
    m["commit_order"] = order.groupby("Issue_ID").cumcount().add(1).reindex(m.index).astype("Int64")
    return m.drop(columns=["first_mid_join"])


def first_commitments(memberships: pd.DataFrame) -> pd.DataFrame:
    """One row per issue that ever had a sprint: its first commitment, and whether it can be used.

    status:
      ok                     the first commitment is known and its sprint was closed
      sprint_unknown         none of the issue's sprints are in the Sprint table
      never_committed        no membership is a commitment (all re-planned, brief, joined at close, ...)
      earlier_sprint_unknown it joined a sprint with no dates before its first known commitment, so that
                             unknown sprint may have been the real first commitment
      sprint_still_open      the first sprint had not been closed when TAWOS was collected
    """
    m = memberships
    issues = pd.Index(m["Issue_ID"].unique(), name="Issue_ID")
    committed = m[m["committed"]]
    first = committed[committed["commit_order"] == 1].set_index("Issue_ID")
    keep = ["JiraID", *SPRINT_COLUMNS, "commitment_time", "entry", "exit"]
    out = first[keep].reindex(issues)
    out.insert(0, "Project_ID", m.groupby("Issue_ID")["Project_ID"].first().reindex(issues))
    out["n_sprints_committed"] = committed.groupby("Issue_ID").size().reindex(issues, fill_value=0)

    resolved = m.groupby("Issue_ID")["Sprint_ID"].count().reindex(issues)
    unknown_joined = m[m["Sprint_ID"].isna()].groupby("Issue_ID")["first_joined"].min().reindex(issues)
    earlier_unknown = unknown_joined < out["commitment_time"]
    out["status"] = np.select(
        [resolved == 0, out["n_sprints_committed"] == 0, earlier_unknown, out["Complete_Date"].isna()],
        ["sprint_unknown", "never_committed", "earlier_sprint_unknown", "sprint_still_open"],
        default="ok",
    )
    return out
