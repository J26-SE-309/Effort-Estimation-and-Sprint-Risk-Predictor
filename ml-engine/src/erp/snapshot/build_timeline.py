"""Build every issue's sprint timeline from TAWOS and report what it shows (ML guide, Phase 1).

Writes two tables to ERP_DATA_DIR/effort-risk/interim/:
  sprint_memberships.parquet  one row per (issue, sprint): committed or not, how it joined and left
  first_commitments.parquet   one row per issue: its first sprint commitment and whether it is usable
and a Markdown report with counts and a few worked examples to check by hand.

Usage:
    erp-build-timeline                      # writes ml-engine/reports/sprint-timeline.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from erp import config, tawos
from erp.explore.profile_tawos import md_table, pct
from erp.snapshot import filters, timeline

EXAMPLE_PROJECT = "MESOS"


def load_stays() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every stay of every issue in a sprint, with the sprint's dates attached."""
    issues = tawos.load("Issue", ["ID", "Issue_Key", "Type", "Story_Point", "Title", "Project_ID", "Sprint_ID",
                                  "Creation_Date", "Resolution_Date"])
    projects = tawos.load("Project", ["ID", "Project_Key", "Repository_ID"])
    sprints = tawos.load("Sprint")
    sprint_log = tawos.load("Change_Log", ["ID", "Issue_ID", "From_Value", "To_Value", "Creation_Date"],
                            filters=pc.field("Field") == "Sprint")

    stays = timeline.replay_sprint_field(sprint_log, issues.set_index("ID")["Creation_Date"])
    stays["source"] = "change_log"
    # Issues with a delivered-in sprint but no Sprint history: in that sprint since they were created.
    only_id = issues[issues["Sprint_ID"].notna() & ~issues["ID"].isin(sprint_log["Issue_ID"])]
    stays = pd.concat([stays, pd.DataFrame({
        "Issue_ID": only_id["ID"].to_numpy(),
        "JiraID": only_id["Sprint_ID"].map(sprints.set_index("ID")["JiraID"]).to_numpy(),
        "joined_at": only_id["Creation_Date"].to_numpy(),
        "left_at": pd.NaT,
        "join_logged": False,
        "source": "issue_sprint_id",
        "Sprint_ID": only_id["Sprint_ID"].astype("Int64").to_numpy(),
    })], ignore_index=True)
    return timeline.resolve_sprints(stays, issues, sprints, projects), issues, projects, sprints


def close_offsets(stays: pd.DataFrame, min_issues: int = 3) -> pd.Series:
    """Clock offset per sprint, in hours: Change_Log time of the bulk move made at close minus Complete_Date.

    Closing a sprint moves its unfinished issues in one go, so several issues leave it within minutes.
    The largest such cluster within 14 hours of Sprint.Complete_Date gives that sprint's offset.
    """
    left = stays[stays["left_at"].notna() & stays["Complete_Date"].notna()]
    hours = (left["left_at"] - left["Complete_Date"]).dt.total_seconds() / 3600
    near = left.assign(hours=hours)[hours.abs() <= 14]
    offsets = {}
    for sprint_id, group in near.groupby("Sprint_ID"):
        x = np.sort(group["hours"].to_numpy())
        best, center, j = 0, np.nan, 0
        for i in range(len(x)):
            while x[i] - x[j] > 10 / 60:
                j += 1
            if i - j + 1 > best:
                best, center = i - j + 1, float(np.median(x[j:i + 1]))
        if best >= min_issues:
            offsets[sprint_id] = center
    return pd.Series(offsets, name="hours")


def share_table(values: pd.Series, labels: dict[str, str]) -> str:
    counts = values.value_counts()
    rows = [{"": text, "Stories": f"{counts.get(key, 0):,}", "Share": pct(counts.get(key, 0), len(values))}
            for key, text in labels.items()]
    return md_table(pd.DataFrame(rows))


def example(memberships: pd.DataFrame, issues: pd.DataFrame, issue_id: int) -> list[str]:
    issue = issues.set_index("ID").loc[issue_id]
    rows = memberships[memberships["Issue_ID"] == issue_id].sort_values(["first_joined", "Start_Date"])
    fmt = "%Y-%m-%d %H:%M"
    table = pd.DataFrame({
        "Sprint": rows["Sprint_Name"].fillna("(not in Sprint table)"),
        "Started": rows["Start_Date"].dt.strftime(fmt).fillna("–"),
        "Closed": rows["Complete_Date"].dt.strftime(fmt).fillna("–"),
        "Joined": rows["first_joined"].dt.strftime(fmt).fillna("–"),
        "Left": rows["last_left"].dt.strftime(fmt).fillna("still in"),
        "Entry": rows["entry"],
        "Exit": rows["exit"].fillna("–"),
        "Order": rows["commit_order"].astype("string").fillna("–"),
    })
    resolved = issue["Resolution_Date"]
    return [
        f"**{issue['Issue_Key']}** ({issue['Type']}, {issue['Story_Point']:g} points; resolved "
        f"{resolved.strftime(fmt) if pd.notna(resolved) else 'never'})", "", md_table(table), "",
    ]


def pick_examples(memberships: pd.DataFrame, commitments: pd.DataFrame, issues: pd.DataFrame,
                  candidates: pd.Series, project_id: int) -> list[tuple[str, int]]:
    """One real story per pattern worth checking by hand, from one project, lowest issue ID first."""
    ok = commitments[(commitments["status"] == "ok") & (commitments["Project_ID"] == project_id)]
    ok = ok[ok.index.isin(candidates[candidates].index)].sort_index()
    resolved = issues.set_index("ID")["Resolution_Date"].reindex(ok.index)
    replanned = memberships[memberships["entry"] == "left_before_start"].merge(
        ok["commitment_time"].rename("first_commitment"), left_on="Issue_ID", right_index=True)
    replanned_first = set(replanned.loc[replanned["first_joined"] < replanned["first_commitment"], "Issue_ID"])
    several = ok["n_sprints_committed"] >= 2
    patterns = [
        ("Planned, and finished in its first sprint",
         (ok["n_sprints_committed"] == 1) & (ok["entry"] == "planned") & (resolved <= ok["Complete_Date"])),
        ("Moved on when the sprint closed (replace mode)", several & (ok["exit"] == "left_at_close")),
        ("Carried over while the closed sprint stayed in its field (keep mode)",
         several & (ok["exit"] == "stayed") & (ok["entry"] == "planned")),
        ("Added after the sprint had started", ok["entry"] == "added_mid_sprint"),
        ("Removed while the sprint was running", ok["exit"] == "left_mid_sprint"),
        ("Re-planned to a later sprint before its first commitment", ok.index.isin(replanned_first)),
    ]
    picked: list[tuple[str, int]] = []
    for title, mask in patterns:
        choices = [i for i in ok.index[np.asarray(mask)] if i not in {p for _, p in picked}]
        if choices:
            picked.append((title, int(choices[0])))
    return picked


def build_report(stays, memberships, commitments, issues, projects, sprints, sensitivity) -> str:
    key_of = projects.set_index("ID")["Project_Key"]
    cand_mask = filters.story_candidates(issues)
    candidates = pd.Series(cand_mask.to_numpy(), index=issues["ID"])
    cand_ids = candidates[candidates].index
    status = commitments["status"].reindex(cand_ids).fillna("no_sprint")
    ok = commitments.loc[status[status == "ok"].index]

    out = ["# Sprint timelines", ""]
    out += [
        "Generated by `erp-build-timeline`. Rebuilds which sprints each issue was in, and when, from the "
        "`Change_Log` history of the Sprint field (`erp/snapshot/timeline.py` explains the rules). "
        "This is step 1 of the data foundation (ML guide, Phase 1); counts only, no issue text.", "",
        f"- Story types: {', '.join(filters.STORY_TYPES)}; story points {filters.SP_RANGE[0]}–{filters.SP_RANGE[1]} "
        "(final value for now; the snapshot step re-checks the value at commitment); non-empty title.",
        f"- Clock tolerance at sprint boundaries: **{timeline.CLOCK_TOLERANCE.total_seconds() / 3600:g} hours** "
        "(section 1).", "",
    ]

    # ---- 1. Clock check ---------------------------------------------------------------------
    offsets = close_offsets(stays)
    bands = pd.cut(offsets.round().abs(), [-0.5, 0.5, 1.5, 5.5, 9.5, 14.5],
                   labels=["0 h", "1 h", "2–5 h", "6–9 h", "10–14 h"])
    sprint_project = sprints.set_index("ID")["Project_ID"].map(key_of)
    per_project = offsets.groupby(sprint_project.reindex(offsets.index)).agg(
        sprints="size", median="median", far=lambda s: (s.round().abs() >= 5).mean())
    far = per_project[(per_project["far"] >= 0.5) & (per_project["sprints"] >= 5)].sort_values("far", ascending=False)
    far_text = ", ".join(f"{key} (median {median:+.0f} h)" for key, median in far["median"].items())
    copies = sprints.assign(Repository_ID=sprints["Project_ID"].map(projects.set_index("ID")["Repository_ID"]))
    copies = copies.groupby(["Repository_ID", "JiraID"])["Start_Date"].agg(["size", "nunique"])
    copies = copies[copies["size"] > 1]
    out += [
        "## 1. Are sprint dates and change times on the same clock?", "",
        "Closing a sprint moves its unfinished issues in one go, so the Change_Log time of that bulk move "
        "should equal `Sprint.Complete_Date`. For the "
        f"{len(offsets):,} sprints with a clear bulk move (3+ issues within 10 minutes, within 14 hours of the "
        "recorded close), the gap between the two is:", "",
        md_table(bands.value_counts(sort=False).rename_axis("Gap").reset_index(name="Sprints")), "",
        f"Most gaps are 0 or 1 hour (daylight saving). Projects where most gaps are 5 hours or more: "
        f"{far_text}. "
        f"Copies of one sprint stored under two projects also have different start dates in "
        f"{int((copies['nunique'] > 1).sum())} of {len(copies)} cases (by exactly 1 hour). "
        "So an issue that joins a sprint within the tolerance after its recorded start counts as planned, and one "
        "that leaves within the tolerance of its recorded close counts as moved on at the close.", "",
    ]

    # ---- 2. Overall -------------------------------------------------------------------------
    resolved = memberships["Sprint_ID"].notna()
    out += [
        "## 2. All issues", "",
        f"- Stays in a sprint replayed: {len(stays):,} for {stays.Issue_ID.nunique():,} issues "
        f"({(stays.source == 'issue_sprint_id').sum():,} of them only from `Issue.Sprint_ID`, with no Sprint "
        "history in the log).",
        f"- (Issue, sprint) memberships: {len(memberships):,}; the sprint is in the `Sprint` table for "
        f"{pct(resolved.sum(), len(memberships))}. Unresolved sprint IDs are mostly the projects with no "
        "`Sprint` rows at all (SERVER, EVG, DNN).", "",
        share_table(memberships["entry"], {
            "planned": "In the sprint when it started (planned)",
            "added_mid_sprint": "Added after the sprint started",
            "brief_stay": "In and out again within a day while the sprint ran (a correction)",
            "joined_at_close": "Added while the sprint was being closed",
            "left_before_start": "Re-planned away before the sprint started (not committed)",
            "joined_after_close": "Added after the sprint was closed (history edit)",
            "sprint_not_started": "Sprint never started (future sprint)",
            "sprint_bad_dates": "Sprint closed before it started (bad dates)",
            "sprint_unknown": "Sprint not in the Sprint table",
        }).replace("| Stories |", "| Memberships |"), "",
    ]

    # ---- 3. Story candidates ----------------------------------------------------------------
    out += [
        "## 3. Can each story's first commitment be found?", "",
        f"Story candidates: **{len(cand_ids):,}**.", "",
        share_table(status, {
            "ok": "Yes: first commitment found, and that sprint was closed",
            "no_sprint": "Never in any sprint",
            "sprint_unknown": "Its sprints are not in the Sprint table (no dates)",
            "earlier_sprint_unknown": "Joined an undated sprint before its first known commitment",
            "never_committed": "Never really committed (re-planned, in and out within a day, added at close)",
            "sprint_still_open": "First sprint was still open when TAWOS was collected",
        }), "",
        f"**{len(ok):,} stories** have a usable first commitment. These are the rows the snapshot and the "
        "labels are built on.", "",
    ]

    # ---- 4. First commitments ---------------------------------------------------------------
    n = ok["n_sprints_committed"].clip(upper=4).map({1: "1", 2: "2", 3: "3", 4: "4 or more"})
    out += [
        "## 4. What happened around the first commitment (usable stories)", "",
        "How it entered its first sprint:", "",
        share_table(ok["entry"], {"planned": "Planned (in the sprint at its start)",
                                  "added_mid_sprint": "Added after the start"}), "",
        "How it left its first sprint:", "",
        share_table(ok["exit"], {
            "stayed": "Stayed (finished there, or kept the closed sprint in its field)",
            "left_at_close": "Moved on when the sprint closed",
            "left_mid_sprint": "Removed while the sprint was running",
            "left_after_close": "Removed later from the closed sprint",
        }), "",
        "Number of sprints it was committed to:", "",
        share_table(n, {"1": "1", "2": "2", "3": "3", "4 or more": "4 or more"}), "",
        "Being in more than one sprint is the raw signal for carry-over (R5). Whether the story was actually "
        "unfinished at the end of its first sprint (R1, R2) is decided in the labels step, using the resolution "
        "date and the status history.", "",
    ]

    # ---- 5. Sensitivity ---------------------------------------------------------------------
    rows = []
    for hours, result in sensitivity.items():
        s = result["status"].reindex(cand_ids).fillna("no_sprint")
        good = result.loc[s[s == "ok"].index]
        rows.append({"Tolerance": f"{hours:g} h", "Usable stories": f"{len(good):,}",
                     "Planned": pct((good.entry == "planned").sum(), len(good)),
                     "Added after start": pct((good.entry == "added_mid_sprint").sum(), len(good)),
                     "In 2+ sprints": pct((good.n_sprints_committed >= 2).sum(), len(good))})
    out += [
        "## 5. Does the tolerance change the picture?", "",
        md_table(pd.DataFrame(rows)), "",
    ]

    # ---- 6. Per project ---------------------------------------------------------------------
    per = pd.DataFrame({"candidates": status.groupby(issues.set_index("ID")["Project_ID"].reindex(cand_ids)).size()})
    grouped = ok.groupby("Project_ID")
    per["usable"] = grouped.size()
    per["planned"] = grouped["entry"].apply(lambda s: (s == "planned").mean())
    per["multi"] = grouped["n_sprints_committed"].apply(lambda s: (s >= 2).mean())
    per["moved_at_close"] = grouped["exit"].apply(lambda s: (s == "left_at_close").mean())
    per = per.fillna(0).sort_values("usable", ascending=False)
    table = pd.DataFrame({
        "Project": per.index.map(key_of), "Candidates": per.candidates.astype(int).map("{:,}".format),
        "Usable": per.usable.astype(int).map("{:,}".format),
        "Planned": per.planned.map(lambda v: f"{100 * v:.0f}%"),
        "In 2+ sprints": per.multi.map(lambda v: f"{100 * v:.0f}%"),
        "Moved on at close": per.moved_at_close.map(lambda v: f"{100 * v:.0f}%"),
        f"≥ {filters.MIN_PROJECT_STORIES} usable": np.where(per.usable >= filters.MIN_PROJECT_STORIES, "yes", ""),
    })
    out += ["## 6. Per project", "", md_table(table), ""]

    # ---- 7. Worked examples -----------------------------------------------------------------
    project_id = int(key_of[key_of == EXAMPLE_PROJECT].index[0])
    out += [
        "## 7. Worked examples to check by hand", "",
        f"Real {EXAMPLE_PROJECT} stories, one per pattern. Times are as stored in TAWOS (sprint dates and change "
        "times may be on different clocks, section 1). Open the issue in the project's public Jira to compare.", "",
    ]
    for title, issue_id in pick_examples(memberships, commitments, issues, candidates, project_id):
        out += [f"### {title}", "", *example(memberships, issues, issue_id)]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path,
                        default=config.REPO_ROOT / "ml-engine" / "reports" / "sprint-timeline.md")
    parser.add_argument("--out-dir", type=Path, default=config.INTERIM_DIR)
    args = parser.parse_args(argv)

    stays, issues, projects, sprints = load_stays()
    memberships = timeline.sprint_memberships(stays)
    commitments = timeline.first_commitments(memberships)
    sensitivity = {hours: timeline.first_commitments(timeline.sprint_memberships(stays, pd.Timedelta(hours=hours)))
                   for hours in (0, 12, 24)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    memberships.to_parquet(args.out_dir / "sprint_memberships.parquet", index=False)
    commitments.reset_index().to_parquet(args.out_dir / "first_commitments.parquet", index=False)

    report = build_report(stays, memberships, commitments, issues, projects, sprints, sensitivity)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"\nWrote {args.report} and {args.out_dir}")


if __name__ == "__main__":
    main()
