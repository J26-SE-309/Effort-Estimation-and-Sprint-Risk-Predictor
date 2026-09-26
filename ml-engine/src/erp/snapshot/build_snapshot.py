"""Build the point-in-time snapshot: one row per story, as it was when it was committed to its first sprint.

Reads interim/first_commitments.parquet (run erp-build-timeline first) and the TAWOS tables, applies the
story filters one after another while logging what each removes (ML guide 6.4), and writes:
  interim/snapshot.parquet   one row per story: the rows every model trains on
  reports/snapshot.md        the filtering log and what the snapshot contains

Values are taken at the snapshot time: for a planned story, the sprint start plus the clock tolerance
(edits made in sprint planning can be logged a few hours "after" the recorded start, see
reports/sprint-timeline.md); for a story added mid-sprint, the moment it was added. Columns starting with
later_ say whether a value changed after that moment. They exist only for sensitivity analyses and must
never be used as features.

Usage:
    erp-build-snapshot                      # writes ml-engine/reports/snapshot.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from erp import config, tawos
from erp.explore.profile_tawos import md_table, pct
from erp.snapshot import filters, history, text, timeline

# snapshot column -> (Change_Log field, Issue column). Title and description are loaded separately.
FIELDS = {
    "issue_type": ("issuetype", "Type"),
    "priority": ("priority", "Priority"),
    "story_points": ("Story Points", "Story_Point"),
    "resolution": ("resolution", "Resolution"),
    "status": ("status", "Status"),
    "title": ("summary", "Title"),
}


def load_commitments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run erp-build-timeline first.")
    commitments = pd.read_parquet(path).set_index("Issue_ID")
    return commitments.rename(columns={"status": "commitment_status", "Project_ID": "commitment_project"})


# What happens within this long after the snapshot time is part of the act of committing the story:
# an estimate typed in a few minutes after the story was added counts as its estimate at commitment, and a
# story resolved within the hour was sprint bookkeeping (added to record finished work), not a commitment.
COMMIT_GRACE = pd.Timedelta(hours=1)
NO_GRACE = pd.Timedelta(0)
GRACE_FIELDS = ("story_points", "resolution")


def snapshot_times(frame: pd.DataFrame) -> pd.Series:
    planned = frame["entry"] == "planned"
    return (frame["Start_Date"] + timeline.CLOCK_TOLERANCE).where(planned, frame["commitment_time"])


def field_at(field: str, column: str, frame: pd.DataFrame, current: pd.DataFrame,
             grace: pd.Timedelta = NO_GRACE) -> pd.DataFrame:
    ids = frame.index.tolist()
    changes = tawos.load("Change_Log", ["ID", "Issue_ID", "Creation_Date", "From_String", "To_String"],
                         filters=(pc.field("Field") == field) & pc.field("Issue_ID").isin(ids))
    return history.value_at(changes, frame["snapshot_time"] + grace, current[column])


def build(commitments: pd.DataFrame) -> tuple[pd.DataFrame, filters.FilterLog, pd.DataFrame]:
    issues = tawos.load("Issue", ["ID", "Issue_Key", "Project_ID", "Type", "Priority", "Story_Point", "Title",
                                  "Resolution", "Status", "Creation_Date"]).set_index("ID")
    projects = tawos.load("Project", ["ID", "Project_Key", "Repository_ID"]).set_index("ID")
    sprint_use = filters.projects_using_sprints(issues, commitments)
    sprint_use["Project_Key"] = sprint_use.index.map(projects["Project_Key"])

    log = filters.FilterLog(issues.join(commitments), "All issues in TAWOS")
    log.keep(log.frame["commitment_status"].notna(), "Was in at least one sprint")
    frame = log.keep(log.frame["commitment_status"] == "ok",
                     "First sprint commitment found (that sprint has reliable dates and was closed)")
    frame = frame.assign(snapshot_time=snapshot_times(frame))

    at = {name: field_at(field, column, frame, issues, COMMIT_GRACE if name in GRACE_FIELDS else NO_GRACE)
          for name, (field, column) in FIELDS.items()}
    frame = frame.assign(**{f"{name}_at": result["value"] for name, result in at.items()},
                         **{f"later_{name}_changed": result["changed_later"] for name, result in at.items()})
    frame["story_points_at"] = pd.to_numeric(frame["story_points_at"], errors="coerce")
    # Today's value comes from the Issue table, which is CSV-quoted; values from the log are not.
    frame["title_at"] = [text.clean_text(value, quoted=not from_log)
                         for value, from_log in zip(frame["title_at"], at["title"]["from_log"], strict=True)]
    log.frame = frame

    log.keep(frame["issue_type_at"].isin(filters.STORY_TYPES),
             f"Story type when committed ({', '.join(filters.STORY_TYPES)})")
    log.keep(log.frame["story_points_at"].notna() | log.frame["Story_Point"].notna(), "Was ever estimated")
    log.keep(log.frame["story_points_at"].notna(), "Already estimated when committed (not only later)")
    log.keep(log.frame["story_points_at"].between(*filters.SP_RANGE),
             f"Story points when committed between {filters.SP_RANGE[0]} and {filters.SP_RANGE[1]}")
    log.keep(log.frame["title_at"] != "", "Non-empty title")
    resolved = log.frame["resolution_at"].fillna("").astype(str).str.strip() != ""
    log.keep(~resolved, "Not resolved when committed or within the hour after")

    dropped = sprint_use[~sprint_use["uses_sprints"] & sprint_use.index.isin(log.frame["Project_ID"])]
    dropped = dropped.assign(stories=log.frame["Project_ID"].value_counts().reindex(dropped.index))
    log.keep(log.frame["Project_ID"].isin(sprint_use.index[sprint_use["uses_sprints"]]),
             f"Project really used sprints (≥ {filters.PROJECT_MIN_SPRINTS} sprints, "
             f"≥ {filters.PROJECT_MIN_SPRINT_SHARE:.0%} of its estimated issues)")

    frame = log.frame
    current_description = tawos.load("Issue", ["ID", "Description"],
                                     filters=pc.field("ID").isin(frame.index.tolist())).set_index("ID")
    description = field_at("description", "Description", frame, current_description)
    raw = [value if from_log else tawos.unquote(value)
           for value, from_log in zip(description["value"], description["from_log"], strict=True)]
    frame = frame.assign(description_at=raw, later_description_changed=description["changed_later"])
    frame["description_text"] = frame["description_at"].map(text.clean_text)

    frame = frame.sort_values("snapshot_time")
    key = pd.DataFrame({"project": frame["Project_ID"], "title": frame["title_at"].str.lower(),
                        "description": frame["description_text"].str.lower()})
    duplicate = key.duplicated() & (key["description"] != "")
    log.frame = frame
    frame = log.keep(~duplicate, "Not a copy of an earlier story in the same project (same title and description)")

    counts = frame.groupby("Project_ID").size()
    snapshot = pd.DataFrame({
        "Issue_ID": frame.index,
        "Issue_Key": frame["Issue_Key"],
        "Project_ID": frame["Project_ID"],
        "Project_Key": frame["Project_ID"].map(projects["Project_Key"]),
        "Repository_ID": frame["Project_ID"].map(projects["Repository_ID"]),
        "project_has_own_model": frame["Project_ID"].map(counts) >= filters.MIN_PROJECT_STORIES,
        "Sprint_ID": frame["Sprint_ID"],
        "Sprint_Name": frame["Sprint_Name"],
        "sprint_start": frame["Start_Date"],
        "sprint_end": frame["End_Date"],
        "sprint_closed": frame["Complete_Date"],
        "created": frame["Creation_Date"],
        "commitment_time": frame["commitment_time"],
        "snapshot_time": frame["snapshot_time"],
        "added_mid_sprint": frame["entry"] == "added_mid_sprint",
        "issue_type": frame["issue_type_at"],
        "priority": frame["priority_at"],
        "priority_level": frame["priority_at"].map(tawos.priority_level),
        "status_at_commitment": frame["status_at"],
        "story_points": frame["story_points_at"],
        "title": frame["title_at"],
        "description": frame["description_at"],
        "description_text": frame["description_text"],
        "description_has_code": frame["description_at"].map(text.has_code),
        **{f"later_{name}_changed": frame[f"later_{name}_changed"]
           for name in ["story_points", "title", "description", "issue_type", "priority"]},
    }).reset_index(drop=True)
    return snapshot, log, dropped


def build_report(snapshot: pd.DataFrame, log: filters.FilterLog, dropped: pd.DataFrame,
                 issues_final_sp: pd.Series) -> str:
    n = len(snapshot)
    out = ["# Point-in-time snapshot", ""]
    out += [
        "Generated by `erp-build-snapshot`. One row per story, frozen at the moment it was committed to its first "
        "sprint (the sprint start plus the clock tolerance for planned stories, the moment it was added for "
        "stories added mid-sprint). Every value is replayed from the `Change_Log`, so later edits to story points, "
        "type, priority, title or description do not leak in. This is steps 2 and 3 of the data foundation "
        "(ML guide, Phase 1); counts only, no issue text.", "",
        "## 1. Filtering log", "",
        md_table(log.table()), "",
        "- *Already estimated when committed*: these stories got their story points more than "
        f"{COMMIT_GRACE.total_seconds() / 3600:g} hour after the snapshot time (an estimate entered within that "
        "time counts as part of committing the story). They have no effort answer for M1; they could still be "
        "added to the risk model (M2) later, with M1's prediction in place of the missing estimate (ML guide 4.2).",
        "- *Not resolved when committed or within the hour after*: the story was already done, or was closed "
        "straight after being added, so it was put in the sprint to record finished work (sprint bookkeeping). "
        "There was no commitment to keep, and counting these would make too many stories look safe.",
        "- *Project really used sprints*: dropped "
        + (", ".join(f"{row.Project_Key} ({row.stories:,} {'story' if row.stories == 1 else 'stories'} left; "
                     f"{row.sprints:.0f} dated sprints, {row.share:.0%} of its estimated issues committed)"
                     for row in dropped.itertuples())
           or "none") + ".",
        "",
    ]

    projects = snapshot.groupby("Project_Key").agg(
        stories=("Issue_ID", "size"), planned=("added_mid_sprint", lambda s: 1 - s.mean()),
        own=("project_has_own_model", "first")).sort_values("stories", ascending=False)
    own = projects[projects["own"]]
    out += [
        "## 2. What the snapshot holds", "",
        f"- **{n:,} stories** from {snapshot.Project_Key.nunique()} projects in "
        f"{snapshot.Sprint_ID.nunique():,} sprints.",
        f"- Planned in sprint planning: {pct((~snapshot.added_mid_sprint).sum(), n)}; added after the sprint "
        f"started: {pct(snapshot.added_mid_sprint.sum(), n)} (kept, with `added_mid_sprint` as a feature).",
        f"- Projects with at least {filters.MIN_PROJECT_STORIES} stories (own model possible): {len(own)}, holding "
        f"{pct(own.stories.sum(), n)} of the stories. The rest only go into the pooled model.", "",
        md_table(snapshot["issue_type"].value_counts().rename_axis("Type").reset_index(name="Stories")), "",
        md_table(snapshot["priority_level"].value_counts().rename_axis("Priority level").reset_index(name="Stories")),
        "",
    ]

    sp = snapshot["story_points"]
    q = sp.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
    final = issues_final_sp.reindex(snapshot["Issue_ID"]).to_numpy()
    differs = ~np.isclose(sp.to_numpy(), final, equal_nan=True)
    out += [
        "## 3. Story points when committed (the effort answer for M1)", "",
        f"- Quartiles: 25% = {q[0.25]:g}, median = {q[0.5]:g}, 75% = {q[0.75]:g}; 90% = {q[0.9]:g}, 99% = {q[0.99]:g}.",
        f"- Changed after commitment: {pct(snapshot.later_story_points_changed.sum(), n)}; the value at commitment "
        f"differs from today's `Issue.Story_Point` for {pct(differs.sum(), n)}. Using today's value would leak "
        "re-estimates into training; the stories whose points never changed form the clean subset for the "
        "sensitivity analysis (ML guide 6.3).", "",
        "## 4. Story text when committed (what the encoders read)", "",
        f"- Title changed after commitment: {pct(snapshot.later_title_changed.sum(), n)}; description changed "
        f"after commitment: {pct(snapshot.later_description_changed.sum(), n)}. For these the older text is used.",
        f"- Empty description: {pct((snapshot.description_text == '').sum(), n)}; description with pasted code or "
        f"logs (removed from `description_text`): {pct(snapshot.description_has_code.sum(), n)}.",
        f"- Median description length: {snapshot.description_text.str.len().median():,.0f} characters.", "",
        "## 5. Per project", "",
        md_table(pd.DataFrame({
            "Project": projects.index, "Stories": projects.stories.map("{:,}".format),
            "Planned": projects.planned.map(lambda v: f"{100 * v:.0f}%"),
            f"≥ {filters.MIN_PROJECT_STORIES} stories": np.where(projects.own, "yes", ""),
        })), "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path,
                        default=config.REPO_ROOT / "ml-engine" / "reports" / "snapshot.md")
    parser.add_argument("--out-dir", type=Path, default=config.INTERIM_DIR)
    args = parser.parse_args(argv)

    snapshot, log, dropped = build(load_commitments(args.out_dir / "first_commitments.parquet"))
    snapshot.to_parquet(args.out_dir / "snapshot.parquet", index=False)
    final_sp = tawos.load("Issue", ["ID", "Story_Point"]).set_index("ID")["Story_Point"]
    report = build_report(snapshot, log, dropped, final_sp)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"Wrote {args.report} and {args.out_dir / 'snapshot.parquet'} ({len(snapshot):,} stories)")


if __name__ == "__main__":
    main()
