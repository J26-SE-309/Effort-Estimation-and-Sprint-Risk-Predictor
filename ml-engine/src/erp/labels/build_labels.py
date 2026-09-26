"""Work out the risk answers for M2: which of the six warning signs R1-R6 happened in each story's first sprint.

Reads interim/snapshot.parquet and interim/sprint_memberships.parquet (run erp-build-timeline and
erp-build-snapshot first) and writes:
  interim/labels.parquet   one row per snapshot story: r1..r6, at_risk, risk_level and the evidence behind them
  reports/labels.md        how often each rule fires, how the rules overlap, and at-risk counts per project

The rules follow proposal Appendix B and ML guide 6.3. "Done" means the issue's resolution was set (TAWOS
records every resolution change, and the last one always equals Issue.Resolution_Date). Sprint boundaries
use the same clock tolerance as the timeline. Every rule is kept as its own column, so merging R1 and R2 or
changing a threshold after the 200-story review needs no new data work.

Usage:
    erp-build-labels                        # writes ml-engine/reports/labels.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from erp import config, tawos
from erp.explore.profile_tawos import md_table, pct
from erp.labels import rules
from erp.snapshot import build_snapshot, history, timeline

RULES = {
    "r1": "R1 Spillover: not done by the end of the sprint",
    "r2": "R2 Delayed closure: done only after the sprint ended, still counted in it",
    "r3": "R3 Re-estimation: story points changed during the sprint",
    "r4": f"R4 Blocked: an open blocker for more than {rules.BLOCKED_SHARE:.0%} of the sprint",
    "r5": "R5 Carry-over: carried into the next sprint unfinished",
    "r6": "R6 Reopening: reopened after being done, in the sprint or the next one",
}


def load(ids: list[int], field: str, columns: list[str]) -> pd.DataFrame:
    return tawos.load("Change_Log", ["ID", "Issue_ID", "Creation_Date", *columns],
                      filters=(pc.field("Field") == field) & pc.field("Issue_ID").isin(ids))


def is_set(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip() != ""


def build(snapshot: pd.DataFrame, memberships: pd.DataFrame,
          tolerance: pd.Timedelta = timeline.CLOCK_TOLERANCE) -> pd.DataFrame:
    s = snapshot.set_index("Issue_ID")
    ids = s.index.tolist()
    close, start = s["sprint_closed"], s["sprint_start"]
    length = close - start

    committed = memberships[memberships["committed"]].set_index("Issue_ID")
    second = committed.loc[committed["commit_order"] == 2, "commitment_time"].reindex(s.index)
    first_exit = committed.loc[committed["commit_order"] == 1, "exit"].reindex(s.index)

    resolution = load(ids, "resolution", ["From_String", "To_String"])
    current = tawos.load("Issue", ["ID", "Resolution"], filters=pc.field("ID").isin(ids)).set_index("ID")["Resolution"]

    def resolved_at(moment: pd.Series) -> pd.Series:
        """Whether each story was resolved at that moment (False where there is no moment)."""
        known = moment.dropna()
        return is_set(history.value_at(resolution, known, current)["value"]).reindex(moment.index, fill_value=False)

    done_events = resolution[is_set(resolution["To_String"])]
    first_done = rules.first_event_after(done_events, s["snapshot_time"])
    # Carried over (R5): committed to the following sprint ("in a row") while not resolved.
    carried = second.notna() & (second <= close + length) & ~resolved_at(second)
    # Done by the end (not R1): resolved when the sprint closed, and not moved on unfinished at the close.
    done_in_sprint = resolved_at(close + tolerance) & ~(carried & (second <= close + tolerance))
    late_done = rules.first_event_after(done_events, close + tolerance)

    points = load(ids, "Story Points", ["From_String", "To_String"])
    old = pd.to_numeric(points["From_String"], errors="coerce")
    new = pd.to_numeric(points["To_String"], errors="coerce")
    points = points[~((old == new) | (old.isna() & new.isna()))]
    points = points.merge(pd.DataFrame({
        "lo": s["snapshot_time"] + build_snapshot.COMMIT_GRACE,
        "hi": (close + tolerance).where(second.isna() | (second > close + tolerance), second),
    }), left_on="Issue_ID", right_index=True)
    reestimates = points[(points["Creation_Date"] > points["lo"]) & (points["Creation_Date"] <= points["hi"])]
    reestimates = reestimates.groupby("Issue_ID").size().reindex(s.index, fill_value=0)

    issue_table = tawos.load("Issue", ["ID", "Issue_Key", "Creation_Date"])
    links = rules.blocking_links(load(ids, "Link", ["From_String", "To_String"]),
                                 issue_table.drop_duplicates("Issue_Key").set_index("Issue_Key")["ID"])
    blocker_ids = links["Blocker_ID"].dropna().astype(int).unique().tolist()
    blocker_open = rules.open_periods(load(blocker_ids, "resolution", ["To_String"]),
                                      issue_table.set_index("ID")["Creation_Date"].reindex(blocker_ids))
    window_end = close.where(first_done.isna() | (first_done > close), first_done)
    blocked = rules.blocked_time(links, blocker_open, pd.DataFrame({"start": s["commitment_time"], "end": window_end}))
    blocked_share = blocked.dt.total_seconds() / length.dt.total_seconds()

    status = load(ids, "status", ["To_String"])
    reopen_events = pd.concat([
        status.loc[status["To_String"].fillna("").str.lower() == "reopened", ["Issue_ID", "Creation_Date"]],
        resolution.loc[is_set(resolution["From_String"]) & ~is_set(resolution["To_String"]),
                       ["Issue_ID", "Creation_Date"]],
    ])
    reopened = rules.first_event_after(reopen_events, first_done.dropna()).reindex(s.index)

    labels = pd.DataFrame({
        "Issue_Key": s["Issue_Key"], "Project_Key": s["Project_Key"], "Sprint_ID": s["Sprint_ID"],
        "added_mid_sprint": s["added_mid_sprint"],
        "first_done_at": first_done, "done_in_sprint": done_in_sprint,
        "second_commitment_at": second, "first_sprint_exit": first_exit,
        "reestimates": reestimates, "blocked_share": blocked_share.round(3),
        "blockers": links[links["Blocker_ID"].notna()].groupby("Issue_ID").size().reindex(s.index, fill_value=0),
        "reopened_at": reopened,
    })
    labels["r1"] = ~done_in_sprint
    labels["r2"] = (labels["r1"] & ~carried & (first_exit == "stayed") & late_done.notna()
                    & (second.isna() | (late_done <= second)))
    labels["r3"] = reestimates > 0
    labels["r4"] = blocked_share > rules.BLOCKED_SHARE
    labels["r5"] = carried
    labels["r6"] = reopened.notna() & (reopened <= close + length)
    labels["n_rules"] = labels[list(RULES)].sum(axis=1)
    labels["at_risk"] = labels["n_rules"] > 0
    labels["risk_level"] = rules.risk_level(labels["n_rules"])
    return labels.reset_index()


def build_report(labels: pd.DataFrame, r1_without_tolerance: float) -> str:
    n = len(labels)
    at_risk = labels["at_risk"]
    out = ["# Risk labels (R1–R6)", ""]
    out += [
        "Generated by `erp-build-labels`. For every snapshot story, which of the six warning signs from proposal "
        "Appendix B happened in its first sprint. A story is *at risk* (M2's answer) when any rule fires; the "
        "risk level is low / medium / high for 0 / 1 / 2+ rules. These are automatic labels: they still need the "
        "200-story human check (ML guide 6.3) before anything is trained on them.", "",
        "## 1. The rules as implemented", "",
        md_table(pd.DataFrame({
            "Rule": list(RULES.values()),
            "How it is computed": [
                "The story was not resolved when the sprint closed (+ clock tolerance), replaying the resolution "
                "history, so a story resolved and then reopened before the close counts; or it was moved on to the "
                "next sprint unresolved at the close.",
                "R1, and the story was resolved later while still recorded in that sprint (not carried over).",
                "A `Story Points` change that alters the value, after the commitment estimate and before the "
                "sprint closed (or before the story moved to its next sprint).",
                "Time with an 'is blocked by' / 'depends on' link to an unresolved issue, while the story was in "
                "the sprint and unresolved, divided by the sprint length. Link dates come from the `Change_Log`.",
                "The story was committed to the following sprint (starting before the end of one more sprint "
                "length) while it was unresolved.",
                "Status set to Reopened, or the resolution cleared, after it was first resolved and before the "
                "end of the following sprint (close + one sprint length).",
            ],
        })), "",
    ]

    counts = pd.DataFrame({
        "Rule": list(RULES.values()),
        "Stories": [f"{labels[r].sum():,}" for r in RULES],
        "Share": [pct(labels[r].sum(), n) for r in RULES],
    })
    levels = labels["risk_level"].value_counts().reindex(["low", "medium", "high"]).fillna(0).astype(int)
    out += [
        "## 2. How often each rule fires", "",
        f"Stories: {n:,}. **At risk: {at_risk.sum():,} ({pct(at_risk.sum(), n)})**.", "",
        md_table(counts), "",
        "Risk level: " + ", ".join(f"{level} {count:,} ({pct(count, n)})" for level, count in levels.items()) + ".",
        "",
        f"R1 without the clock tolerance (resolved by the recorded close exactly): {r1_without_tolerance:.1%}, "
        f"against {labels['r1'].mean():.1%} with it.", "",
    ]

    names = [r.upper() for r in RULES]
    overlap = pd.DataFrame(index=names, columns=names, dtype=object)
    for a in RULES:
        for b in RULES:
            base = labels[a].sum()
            overlap.loc[a.upper(), b.upper()] = pct((labels[a] & labels[b]).sum(), base) if base else "–"
    out += [
        "## 3. How the rules overlap", "",
        "Share of the stories where the row's rule fires that also have the column's rule. R2 lies inside R1 by "
        "definition, and R5 largely does too; the 200-story review decides whether to merge them. Because of "
        "this overlap, the rule count behind the risk level partly counts the same failure twice.", "",
        md_table(overlap.reset_index().rename(columns={"index": "Rule"})), "",
    ]

    by_entry = labels.groupby("added_mid_sprint")["at_risk"].agg(["size", "mean"])
    out += [
        "## 4. Planned stories and stories added mid-sprint", "",
        f"- Planned: {by_entry.loc[False, 'size']:,} stories, {by_entry.loc[False, 'mean']:.1%} at risk.",
        f"- Added after the sprint started: {by_entry.loc[True, 'size']:,} stories, "
        f"{by_entry.loc[True, 'mean']:.1%} at risk.", "",
    ]

    per = labels.groupby("Project_Key").agg(stories=("at_risk", "size"), at_risk=("at_risk", "sum"))
    per["share"] = per["at_risk"] / per["stories"]
    per["not_at_risk"] = per["stories"] - per["at_risk"]
    per = per.sort_values("stories", ascending=False)
    minimum = 50
    table = pd.DataFrame({
        "Project": per.index, "Stories": per.stories.map("{:,}".format), "At risk": per.at_risk.map("{:,}".format),
        "Share": per.share.map(lambda v: f"{100 * v:.0f}%"),
        "Own risk model?": np.where((per.at_risk >= minimum) & (per.not_at_risk >= minimum), "yes", ""),
    })
    out += [
        "## 5. Per project", "",
        f"The risk model learns mainly from the rarer class, so a project gets its own risk model only with at "
        f"least {minimum} at-risk and {minimum} not-at-risk stories (a suggested line, ML guide 6.5); the others "
        "go into the pooled model.", "",
        md_table(table), "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=config.REPO_ROOT / "ml-engine" / "reports" / "labels.md")
    parser.add_argument("--out-dir", type=Path, default=config.INTERIM_DIR)
    args = parser.parse_args(argv)

    for name in ("snapshot.parquet", "sprint_memberships.parquet"):
        if not (args.out_dir / name).exists():
            raise FileNotFoundError(f"{args.out_dir / name} not found. Run erp-build-timeline and erp-build-snapshot.")
    snapshot = pd.read_parquet(args.out_dir / "snapshot.parquet")
    memberships = pd.read_parquet(args.out_dir / "sprint_memberships.parquet")
    labels = build(snapshot, memberships)
    strict = build(snapshot, memberships, pd.Timedelta(0))
    labels.to_parquet(args.out_dir / "labels.parquet", index=False)

    report = build_report(labels, strict["r1"].mean())
    args.report.write_text(report, encoding="utf-8")
    print(f"Wrote {args.report} and {args.out_dir / 'labels.parquet'} ({labels['at_risk'].mean():.1%} at risk)")


if __name__ == "__main__":
    main()
