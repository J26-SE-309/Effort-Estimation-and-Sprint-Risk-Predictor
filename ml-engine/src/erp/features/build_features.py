"""Compute every feature in erp.features.catalog for each snapshot story (ML guide Phase 1, step 4).

Reads interim/snapshot.parquet, sprint_memberships.parquet and labels.parquet and writes:
  interim/features.parquet   one row per story: Issue_ID and every catalogue feature
  reports/features.md        what each feature means, how complete it is, and how it differs between
                             at-risk and not-at-risk stories (descriptive only, not a model result)

Labels feed only the two team-history rates, and only for past stories whose outcome was already known,
so rerun this after erp-build-labels whenever the rules change.

Usage:
    erp-build-features                      # writes ml-engine/reports/features.md
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from erp import config, tawos
from erp.explore.profile_tawos import md_table, pct
from erp.features import catalog, team, text
from erp.labels import rules
from erp.snapshot import build_snapshot, filters, history, timeline

TEST_TYPES = {"Test", "Test Task", "Test Case"}
TEST_TITLE = re.compile(r"\btest(?:s|ing)?\b", re.I)
BLOCKS = ("blocks", "is depended on by", "has to be done before")
DAY = pd.Timedelta(days=1)


def load_log(field: str, ids: list[int]) -> pd.DataFrame:
    return tawos.load("Change_Log", ["ID", "Issue_ID", "Creation_Date", "From_String", "To_String"],
                      filters=(pc.field("Field") == field) & pc.field("Issue_ID").isin(ids))


def is_set(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip() != ""


def link_features(stories: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    """Dependency and traceability clues from the links that existed at each story's snapshot time."""
    ids = stories.index.tolist()
    key_to_id = pd.Series(issues.index, index=issues["Issue_Key"]).groupby(level=0).first()
    links = history.link_periods(load_log("Link", ids), key_to_id)
    links = links.merge(stories["snapshot_time"].rename("t"), left_on="Issue_ID", right_index=True)
    exists = (links["added_at"] <= links["t"]) & (links["removed_at"].isna() | (links["removed_at"] > links["t"]))
    active = links[exists]

    depends = active[active["phrase"].isin(rules.DEPENDS_ON)]
    known = depends[depends["Target_ID"].notna()]
    pairs = pd.DataFrame({"Issue_ID": known["Target_ID"].astype(int).to_numpy(), "at": known["t"].to_numpy()})
    resolved = is_set(history.values_at(load_log("resolution", pairs["Issue_ID"].unique().tolist()), pairs,
                                        issues["Resolution"]))
    open_blockers = known[~resolved.to_numpy()]

    target_type = active["Target_ID"].map(issues["Type"])
    target_title = active["Target_ID"].map(issues["Title"]).fillna("")
    tests = active[target_type.isin(TEST_TYPES) | target_title.str.contains(TEST_TITLE)]

    epic = history.values_at(load_log("Epic Link", ids),
                             pd.DataFrame({"Issue_ID": ids, "at": stories["snapshot_time"].to_numpy()}),
                             pd.Series(dtype=object))

    def count(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby("Issue_ID")["target_key"].nunique().reindex(stories.index, fill_value=0)

    return pd.DataFrame({
        "blocker_count": count(open_blockers),
        "dep_out_degree": count(depends),
        "dep_in_degree": count(active[active["phrase"].isin(BLOCKS)]),
        "linked_issue_count": count(active),
        "has_linked_tests": count(tests) > 0,
        "has_epic": is_set(epic).to_numpy(),
    }, index=stories.index)


def sprint_outcomes(memberships: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    """Every committed membership in a closed sprint, with its points and whether it was finished there."""
    m = memberships[memberships["committed"] & memberships["Complete_Date"].notna()
                    & memberships["dates_reliable"]].reset_index(drop=True)
    ids = m["Issue_ID"].unique().tolist()
    points, resolution = load_log("Story Points", ids), load_log("resolution", ids)
    close = m["Complete_Date"] + timeline.CLOCK_TOLERANCE

    def at(log: pd.DataFrame, moments: pd.Series, current: pd.Series) -> pd.Series:
        return history.values_at(log, pd.DataFrame({"Issue_ID": m["Issue_ID"], "at": moments}), current)

    m["points_at_commit"] = pd.to_numeric(at(points, m["commitment_time"] + build_snapshot.COMMIT_GRACE,
                                             issues["Story_Point"]), errors="coerce")
    m["points_at_close"] = pd.to_numeric(at(points, close, issues["Story_Point"]), errors="coerce")
    resolved_before = is_set(at(resolution, m["commitment_time"], issues["Resolution"]))
    resolved_at_close = is_set(at(resolution, close, issues["Resolution"]))
    m["completed"] = ~resolved_before & resolved_at_close & m["in_at_close"]
    return m


def team_features(stories: pd.DataFrame, outcomes: pd.DataFrame, labels: pd.DataFrame,
                  issues: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    sprints = outcomes.assign(done_points=outcomes["points_at_close"].where(outcomes["completed"], 0).fillna(0))
    sprints = sprints.groupby(["Project_ID", "Sprint_ID"]).agg(
        started_at=("Start_Date", "first"), closed_at=("Complete_Date", "first"),
        completed_points=("done_points", "sum")).reset_index()
    out = team.previous_sprint_stats(sprints, stories)

    past = labels.merge(snapshot[["Issue_ID", "Project_ID", "sprint_start", "sprint_closed"]], on="Issue_ID")
    past["r1_known"] = past["sprint_closed"] + timeline.CLOCK_TOLERANCE
    past["r6_known"] = past["sprint_closed"] + (past["sprint_closed"] - past["sprint_start"])
    out["historical_spillover_rate"] = team.recent_rate(past, stories, "r1", "r1_known")
    out["reopen_rate"] = team.recent_rate(past, stories, "r6", "r6_known")

    finished = issues[issues["Type"].isin(filters.STORY_TYPES) & issues["Resolution_Date"].notna()
                      & issues["In_Progress_Minutes"].notna()]
    cycle = pd.DataFrame({"Project_ID": finished["Project_ID"], "hours": finished["In_Progress_Minutes"] / 60,
                          "known": finished["Resolution_Date"]})
    out["mean_cycle_time_hours"] = team.recent_rate(cycle, stories, "hours", "known")

    active = np.zeros(len(stories), dtype=int)
    for project, group in stories.groupby("Project_ID"):
        own = sprints[sprints["Project_ID"] == project]
        starts = np.sort(own["started_at"].to_numpy(dtype="datetime64[us]"))
        closes = np.sort(own["closed_at"].to_numpy(dtype="datetime64[us]"))
        t = group["snapshot_time"].to_numpy(dtype="datetime64[us]")
        running = np.searchsorted(starts, t, side="right") - np.searchsorted(closes, t, side="right")
        active[stories.index.get_indexer(group.index)] = np.maximum(running - 1, 0)
    out["parallel_sprints"] = active
    return out


def sprint_features(stories: pd.DataFrame, outcomes: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    """What the story's sprint looked like at its snapshot time: committed points and work in progress."""
    members = outcomes[["Sprint_ID", "Issue_ID", "commitment_time", "last_left", "points_at_commit"]]
    members = members.rename(columns={"Issue_ID": "member"})
    pairs = stories[["Sprint_ID", "snapshot_time"]].reset_index().merge(members, on="Sprint_ID")
    present = pairs[(pairs["commitment_time"] <= pairs["snapshot_time"])
                    & (pairs["last_left"].isna() | (pairs["last_left"] > pairs["snapshot_time"]))]
    # Other stories only: the story's own points are M1's answer, so they must not hide inside a feature.
    others = present[present["member"] != present["Issue_ID"]].reset_index(drop=True)
    committed_points = others.groupby("Issue_ID")["points_at_commit"].sum().reindex(stories.index, fill_value=0)
    moments = pd.DataFrame({"Issue_ID": others["member"], "at": others["snapshot_time"]})
    member_ids = others["member"].unique().tolist()
    status = history.values_at(load_log("status", member_ids), moments, issues["Status"])
    resolved = is_set(history.values_at(load_log("resolution", member_ids), moments, issues["Resolution"]))
    started = team.is_started(status.astype("string"), resolved)
    wip = started.groupby(others["Issue_ID"]).sum().reindex(stories.index, fill_value=0)

    return pd.DataFrame({
        "sprint_length_days": (stories["sprint_end"] - stories["sprint_start"]) / DAY,
        "days_into_sprint": ((stories["commitment_time"] - stories["sprint_start"]) / DAY).where(
            stories["added_mid_sprint"], 0.0).clip(lower=0),
        "sprint_committed_points": committed_points,
        "wip_at_commitment": wip,
        "in_progress_at_commitment": team.is_started(stories["status_at_commitment"].astype("string"),
                                                     pd.Series(False, index=stories.index)),
    }, index=stories.index)


def build(snapshot: pd.DataFrame, memberships: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    issues = tawos.load("Issue", ["ID", "Issue_Key", "Project_ID", "Type", "Title", "Story_Point", "Resolution",
                                  "Resolution_Date", "Status", "In_Progress_Minutes"]).set_index("ID")
    stories = snapshot.set_index("Issue_ID")
    words = text.text_features(stories)
    links = link_features(stories, issues)
    outcomes = sprint_outcomes(memberships, issues)
    history_ = team_features(stories, outcomes, labels, issues, snapshot)
    sprint = sprint_features(stories, outcomes, issues)

    features = pd.concat([words, links, history_, sprint], axis=1)
    velocity = features["team_velocity_rolling"]
    features["commitment_to_velocity_ratio"] = features["sprint_committed_points"] / velocity.where(velocity > 0)
    features["invest_independent"] = features["blocker_count"] == 0
    features["invest_valuable"] = features["user_story_format"] | words["states_goal"]
    features["invest_testable"] = (features["has_acceptance_criteria"] | features["has_linked_tests"]
                                   | features["mentions_tests"])
    traces = pd.concat([features["has_epic"], features["linked_issue_count"] > 0, features["has_linked_tests"]], axis=1)
    features["traceability_coverage_pct"] = traces.mean(axis=1).round(4)
    features["unlinked_artifact_count"] = 3 - traces.sum(axis=1)
    features["added_mid_sprint"] = stories["added_mid_sprint"]
    features["story_points"] = stories["story_points"]
    features["issue_type"] = stories["issue_type"]
    features["priority_level"] = stories["priority_level"]
    features["project_key"] = stories["Project_Key"]
    return features[catalog.names()].rename_axis("Issue_ID").reset_index()


def is_text(values: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values)


def describe(values: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(values):
        return f"{values.mean():.1%} yes"
    if is_text(values):
        top = values.value_counts(normalize=True).head(2)
        return ", ".join(f"{k} {v:.0%}" for k, v in top.items())
    return f"median {values.median():,.2f}"


def contrast(values: pd.Series, at_risk: pd.Series) -> str:
    if is_text(values):
        return ""
    risky, safe = values[at_risk].astype(float).mean(), values[~at_risk].astype(float).mean()
    if pd.api.types.is_bool_dtype(values):
        return f"{risky:.1%} vs {safe:.1%}"
    return f"{risky:,.2f} vs {safe:,.2f}"


def build_report(features: pd.DataFrame, labels: pd.DataFrame) -> str:
    at_risk = features[["Issue_ID"]].merge(labels[["Issue_ID", "at_risk"]], on="Issue_ID")["at_risk"].to_numpy()
    at_risk = pd.Series(at_risk, index=features.index)
    rows = [{
        "Group": f.group, "Feature": f"`{f.name}`", "Meaning": f.meaning,
        "Known at commitment because": f.known_because,
        "Filled": pct(features[f.name].notna().sum(), len(features)), "Typical": describe(features[f.name]),
        "At risk vs not": contrast(features[f.name], at_risk),
    } for f in catalog.FEATURES]
    cold = (features["history_sprints"] == 0).sum()
    return "\n".join([
        "# Features", "",
        "Generated by `erp-build-features`. The clues each model sees about a story, all computed from what was "
        "known at the story's snapshot time (the point-in-time rule, ML guide 6.1). The story text itself goes to "
        "the encoders (E1–E4); the numbers here are joined with it. The definitions live in "
        "`erp/features/catalog.py`.", "",
        f"- Stories: {len(features):,}; features: {len(catalog.FEATURES)} in {len(catalog.GROUPS)} groups.",
        "- The `requirement_quality`, `acceptance_criteria` and `traceability` groups are **proxies** for "
        "Components 1–3 (ML guide 6.6, route 2), named as in the backend's `UpstreamSignals` contract so the "
        "teammates' batch scores can replace them. The H2 experiment drops these groups one at a time.",
        f"- Missing values are real gaps, not zeros: {cold:,} stories ({pct(cold, len(features))}) come from a "
        "project's first sprint, so the team-history clues are empty (cold start); rates need at least 5 past "
        "stories.",
        "- *At risk vs not* compares the mean (or share) between at-risk and not-at-risk stories. It is a first "
        "look at which clues carry a signal, not a model result.", "",
        md_table(pd.DataFrame(rows)), "",
    ])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, default=config.REPO_ROOT / "ml-engine" / "reports" / "features.md")
    parser.add_argument("--out-dir", type=Path, default=config.INTERIM_DIR)
    args = parser.parse_args(argv)

    needed = ["snapshot.parquet", "sprint_memberships.parquet", "labels.parquet"]
    missing = [name for name in needed if not (args.out_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {', '.join(missing)}. Run erp-build-timeline, erp-build-snapshot and "
                                "erp-build-labels first.")
    snapshot, memberships, labels = (pd.read_parquet(args.out_dir / name) for name in needed)
    features = build(snapshot, memberships, labels)
    features.to_parquet(args.out_dir / "features.parquet", index=False)
    args.report.write_text(build_report(features, labels), encoding="utf-8")
    print(f"Wrote {args.report} and {args.out_dir / 'features.parquet'} ({len(features):,} stories)")


if __name__ == "__main__":
    main()
