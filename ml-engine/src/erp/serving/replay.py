"""Real TAWOS sprints as sprint records (erp.serving.history), and the parity check that proves the service
computes the team-history features exactly as training did.

The records carry what training used: every story committed to a closed sprint with reliable dates, with its
points and whether it was done there (the sprint's velocity); for the stories training labelled, the spillover
and reopening outcomes (R1, R6) at their first commitment; and the project's other resolved stories, without a
sprint, for the cycle time. They serve as development data for the
service (backend: python -m app.devdata) and as the parity check's input:

    erp-history-parity                   # every project -> ml-engine/reports/history-parity.md
    erp-history-parity --projects MESOS  # some projects only

The check rebuilds each snapshot story's features from the records at its snapshot time and compares them with
interim/features.parquet.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from erp import config, tawos
from erp.features.build_features import sprint_outcomes
from erp.serving import history
from erp.snapshot.filters import STORY_TYPES

COMPARED = ["team_velocity_rolling", "velocity_variance", "history_sprints", "historical_spillover_rate",
            "reopen_rate", "mean_cycle_time_hours", "sprint_length_days", "parallel_sprints"]
TOLERANCE = 1e-6


def records(project_keys: list[str] | None = None, prefix: str = "") -> tuple[pd.DataFrame, pd.DataFrame,
                                                                               pd.DataFrame]:
    """(sprints, items, stories) of these TAWOS projects (all when None), with project ids prefix + key.

    stories are the training snapshot stories of those projects (index Issue_ID; Project_ID, snapshot_time,
    Sprint_ID), for the parity check.
    """
    snapshot = pd.read_parquet(config.INTERIM_DIR / "snapshot.parquet")
    memberships = pd.read_parquet(config.INTERIM_DIR / "sprint_memberships.parquet")
    labels = pd.read_parquet(config.INTERIM_DIR / "labels.parquet")
    key_of = snapshot.drop_duplicates("Project_ID").set_index("Project_ID")["Project_Key"]
    if project_keys:
        unknown = set(project_keys) - set(key_of)
        if unknown:
            raise KeyError(f"not TAWOS projects with snapshot stories: {', '.join(sorted(unknown))}")
        key_of = key_of[key_of.isin(project_keys)]
    memberships = memberships[memberships["Project_ID"].isin(key_of.index)]
    issues = tawos.load("Issue", ["ID", "Issue_Key", "Project_ID", "Type", "Story_Point", "Resolution",
                                  "Resolution_Date", "Status", "In_Progress_Minutes"]).set_index("ID")
    outcomes = sprint_outcomes(memberships, issues)
    project = (prefix + outcomes["Project_ID"].map(key_of)).to_numpy()

    # In training's order (by sprint id), so sprints closing at the same moment fall the same way in "the last 3".
    sprints = pd.DataFrame({
        "project_id": project, "sprint_id": outcomes["Sprint_ID"].astype(str).to_numpy(),
        "name": outcomes["Sprint_Name"].to_numpy(), "started_at": outcomes["Start_Date"].to_numpy(),
        "planned_end": outcomes["End_Date"].to_numpy(), "closed_at": outcomes["Complete_Date"].to_numpy(),
        "order": outcomes["Sprint_ID"].to_numpy(),
    }).drop_duplicates(["project_id", "sprint_id"]).sort_values("order", kind="stable")
    sprints = sprints.drop(columns="order").reset_index(drop=True)

    # Outcomes where training labelled the story: at its first commitment, which is its snapshot sprint.
    labelled = labels.reset_index()[["index", "Issue_ID", "Sprint_ID", "r1", "r6"]].rename(columns={"index": "order"})
    items = outcomes.merge(labelled, on=["Issue_ID", "Sprint_ID"], how="left")
    first = items["commit_order"] == 1
    items = pd.DataFrame({
        "project_id": project, "sprint_id": items["Sprint_ID"].astype(str).to_numpy(),
        "story_id": items["Issue_ID"].map(issues["Issue_Key"]).to_numpy(),
        "issue_type": items["Issue_ID"].map(issues["Type"]).to_numpy(),
        "committed_at": items["commitment_time"].to_numpy(),
        "left_at": items["last_left"].where(~items["in_at_close"]).to_numpy(),
        "points_at_commit": items["points_at_commit"].to_numpy(),
        "points_at_close": items["points_at_close"].to_numpy(),
        "done_in_sprint": items["completed"].to_numpy(),
        "spilled_over": items["r1"].where(first).astype("boolean").to_numpy(),
        "reopened": items["r6"].where(first).astype("boolean").to_numpy(),
        "started_at": pd.NaT, "resolved_at": items["Issue_ID"].map(issues["Resolution_Date"]).to_numpy(),
        "hours_in_progress": (items["Issue_ID"].map(issues["In_Progress_Minutes"]) / 60).to_numpy(),
        # The rates take "the last 50" outcomes, and many share a sprint's close: keep training's order of
        # labelled stories, so ties fall the same way.
        "order": items["order"].to_numpy(),
    }).sort_values("order", kind="stable", na_position="last").drop(columns="order")

    # The project's other resolved stories, never in one of these sprints: only their cycle time counts.
    resolved = issues[issues["Project_ID"].isin(key_of.index) & issues["Type"].isin(STORY_TYPES)
                      & issues["Resolution_Date"].notna() & issues["In_Progress_Minutes"].notna()
                      & ~issues["Issue_Key"].isin(items["story_id"])]
    outside = pd.DataFrame({
        "project_id": (prefix + resolved["Project_ID"].map(key_of)).to_numpy(), "sprint_id": None,
        "story_id": resolved["Issue_Key"].to_numpy(), "issue_type": resolved["Type"].to_numpy(),
        "resolved_at": resolved["Resolution_Date"].to_numpy(),
        "hours_in_progress": (resolved["In_Progress_Minutes"] / 60).to_numpy()})
    items = pd.concat([items, outside.reindex(columns=items.columns)], ignore_index=True)

    own = snapshot[snapshot["Project_ID"].isin(key_of.index)]
    stories = pd.DataFrame({"Project_ID": (prefix + own["Project_ID"].map(key_of)).to_numpy(),
                            "snapshot_time": own["snapshot_time"].to_numpy(),
                            "Sprint_ID": own["Sprint_ID"].astype(str).to_numpy()},
                           index=pd.Index(own["Issue_ID"].to_numpy(), name="Issue_ID"))
    return sprints, items, stories


def parity(project_keys: list[str] | None = None) -> pd.DataFrame:
    """Per feature: stories compared, how many match training within TOLERANCE, and the largest difference."""
    sprints, items, stories = records(project_keys)
    served = history.features(sprints, items, stories)
    trained = pd.read_parquet(config.INTERIM_DIR / "features.parquet").set_index("Issue_ID").reindex(stories.index)
    rows = []
    for name in COMPARED:
        a, b = served[name].astype(float).to_numpy(), trained[name].astype(float).to_numpy()
        both_missing = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=TOLERANCE, atol=TOLERANCE) | both_missing
        diff = np.abs(a - b)
        rows.append({"feature": name, "stories": len(a), "match": int(close.sum()),
                     "match_share": close.mean(), "missing_mismatch": int((np.isnan(a) != np.isnan(b)).sum()),
                     "largest_difference": float(np.nanmax(diff)) if np.isfinite(diff).any() else 0.0,
                     "mean_difference": float(np.nanmean(diff)) if np.isfinite(diff).any() else 0.0})
    return pd.DataFrame(rows)


def report(result: pd.DataFrame, projects: list[str] | None) -> str:
    lines = ["# Team-history parity: the service against training", "",
             "Generated by `erp-history-parity`. The service computes the team-history and sprint-timing features "
             "from sprint records (`erp/serving/history.py`); here the records are the real TAWOS sprints "
             "(`erp/serving/replay.py`), and every snapshot story's features are recomputed at its snapshot time and "
             f"compared with the ones training used (`interim/features.parquet`), within {TOLERANCE:g}.", "",
             f"Projects: {', '.join(projects) if projects else 'all'}.", "",
             "| Feature | Stories | Match | Missing in one only | Largest difference | Mean difference |",
             "|---|---|---|---|---|---|"]
    for r in result.itertuples():
        lines.append(f"| `{r.feature}` | {r.stories:,} | {r.match_share:.2%} | {r.missing_mismatch:,} | "
                     f"{r.largest_difference:.4g} | {r.mean_difference:.4g} |")
    lines += ["", "A rate over \"the last 50\" stories is ambiguous when several stories became known at the same "
              "moment at the edge of the window (a sprint's close; stories resolved together in one bulk change). "
              "The records keep training's order where it is known (labelled stories, sprints), so those ties fall "
              "the same way; the remaining cycle-time differences are all such ties between stories resolved in "
              "the same second."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--projects", nargs="*", help="TAWOS project keys (default: all)")
    parser.add_argument("--report", type=Path, default=config.REPO_ROOT / "ml-engine" / "reports" /
                        "history-parity.md")
    args = parser.parse_args(argv)
    result = parity(args.projects)
    args.report.write_text(report(result, args.projects), encoding="utf-8")
    print(result.to_string(index=False))
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
