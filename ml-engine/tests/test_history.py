"""Team history from sprint records (erp.serving.history): the training definitions, on a team small enough to
check by hand, and the parity check against training on a real TAWOS project."""

import numpy as np
import pandas as pd
import pytest

from erp import config
from erp.serving import history

START = pd.Timestamp("2026-01-05")
DAY = pd.Timedelta(days=1)


@pytest.fixture
def team() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seven closed two-week sprints finishing 10, 20, ... 70 points, each with one story spilled over; then two
    sprints running side by side; and resolved stories outside any sprint."""
    sprints, items = [], []
    for k in range(7):
        start = START + 14 * k * DAY
        sprints.append({"project_id": "P", "sprint_id": f"S{k + 1}", "name": f"Sprint {k + 1}", "started_at": start,
                        "planned_end": start + 14 * DAY, "closed_at": start + 14 * DAY})
        items += [
            {"project_id": "P", "sprint_id": f"S{k + 1}", "story_id": f"D{k + 1}", "committed_at": start,
             "points_at_commit": 10 * (k + 1), "points_at_close": 10 * (k + 1), "done_in_sprint": True,
             "spilled_over": False, "reopened": k == 0},
            {"project_id": "P", "sprint_id": f"S{k + 1}", "story_id": f"L{k + 1}", "committed_at": start,
             "points_at_commit": 5, "points_at_close": 5, "done_in_sprint": False, "spilled_over": True,
             "reopened": False},
        ]
    now = START + 98 * DAY  # sprint 8 and B1 start when sprint 7 closes
    for sprint_id in ("S8", "B1"):
        sprints.append({"project_id": "P", "sprint_id": sprint_id, "started_at": now, "planned_end": now + 14 * DAY})
    items += [
        {"project_id": "P", "sprint_id": "S8", "story_id": "W1", "committed_at": now, "started_at": now + DAY},
        {"project_id": "P", "sprint_id": "S8", "story_id": "W2", "committed_at": now, "started_at": now + DAY,
         "resolved_at": now + 1.5 * DAY},
        {"project_id": "P", "sprint_id": "S8", "story_id": "W3", "committed_at": now},
        {"project_id": "P", "sprint_id": "S8", "story_id": "W4", "committed_at": now, "left_at": now + DAY,
         "started_at": now},
    ]
    items += [{"project_id": "P", "story_id": f"C{h}", "issue_type": "Story", "resolved_at": START + h * DAY,
               "hours_in_progress": h} for h in (10, 20, 30, 40, 50)]
    items.append({"project_id": "P", "story_id": "E1", "issue_type": "Epic", "resolved_at": START + DAY,
                  "hours_in_progress": 1000})
    return pd.DataFrame(sprints), pd.DataFrame(items)


def _at(sprints, items, moment, sprint_id="S8", exclude=frozenset()) -> pd.Series:
    stories = pd.DataFrame({"Project_ID": ["P"], "snapshot_time": [moment], "Sprint_ID": [sprint_id]})
    return history.features(sprints, items, stories, exclude).iloc[0]


def test_velocity_counts_closed_sprints_only(team):
    first_close = START + 14 * DAY
    before = _at(*team, first_close - pd.Timedelta(minutes=1))
    assert before["history_sprints"] == 0 and np.isnan(before["team_velocity_rolling"])
    after = _at(*team, START + 100 * DAY)
    assert after["history_sprints"] == 7
    assert after["team_velocity_rolling"] == 60  # the last 3: 50, 60, 70 (the spilled stories finished nothing)
    assert after["velocity_variance"] == pytest.approx(np.var([30, 40, 50, 60, 70], ddof=1))


def test_outcomes_count_only_once_they_could_be_known(team):
    third_close = START + 42 * DAY
    # Spillover is known 12 h after the close: before that, sprint 3's two stories do not count, and four
    # outcomes are too few (at least 5).
    assert np.isnan(_at(*team, third_close + pd.Timedelta(hours=11))["historical_spillover_rate"])
    assert _at(*team, third_close + pd.Timedelta(hours=13))["historical_spillover_rate"] == 0.5
    # A reopening is known one sprint length after the close: at the end, 14 outcomes, one reopened.
    assert np.isnan(_at(*team, third_close + pd.Timedelta(hours=13))["reopen_rate"])
    assert _at(*team, START + 200 * DAY)["reopen_rate"] == pytest.approx(1 / 14)


def test_cycle_time_uses_resolved_stories_of_the_story_types(team):
    assert _at(*team, START + 100 * DAY)["mean_cycle_time_hours"] == 30  # the epic's 1,000 h does not count


def test_sprint_timing_and_work_in_progress(team):
    now = START + 100 * DAY  # two days into sprint 8
    own = _at(*team, now)
    assert own["sprint_length_days"] == 14 and own["parallel_sprints"] == 1  # B1 runs beside it
    assert own["wip_at_commitment"] == 1  # W1; W2 is resolved, W3 not started, W4 left the sprint
    assert _at(*team, now, exclude=frozenset({"W1"}))["wip_at_commitment"] == 0
    planned = _at(*team, now, sprint_id="S9")  # a sprint being planned is not in the records yet
    assert planned["parallel_sprints"] == 2 and np.isnan(planned["sprint_length_days"])


def test_context_in_the_api_field_names(team):
    team_context, sprint_context = history.context(*team, "P", START + 100 * DAY, sprint_id="S8")
    assert team_context["velocity_mean"] == 60 and team_context["closed_sprints"] == 7
    assert team_context["spillover_rate"] == 0.5 and team_context["mean_cycle_time_hours"] == 30
    assert sprint_context == {"length_days": 14.0, "parallel_sprints": 1, "wip": 1}
    empty_team, empty_sprint = history.context(pd.DataFrame(), pd.DataFrame(), "NEW", START)
    assert empty_team["velocity_mean"] is None and empty_sprint["length_days"] is None


@pytest.mark.skipif(not (config.INTERIM_DIR / "features.parquet").exists(), reason="needs the training data")
def test_parity_with_training_on_a_real_project():
    from erp.serving import replay

    result = replay.parity(["INDY"]).set_index("feature")
    exact = result.drop(index="mean_cycle_time_hours")
    assert (exact["match_share"] == 1).all(), exact

    # Cycle time may differ only where stories resolved in the same second meet at the edge of "the last 50".
    from erp.snapshot.filters import STORY_TYPES

    sprints, items, stories = replay.records(["INDY"])
    served = history.features(sprints, items, stories)["mean_cycle_time_hours"]
    trained = pd.read_parquet(config.INTERIM_DIR / "features.parquet").set_index("Issue_ID")
    trained = trained["mean_cycle_time_hours"].reindex(stories.index)
    differ = stories[~np.isclose(served, trained, rtol=1e-6, atol=1e-6, equal_nan=True)]
    _, resolved = history.prepare(sprints, items)
    resolved = resolved.drop_duplicates("story_id", keep="last")
    resolved = np.sort(resolved.loc[resolved["hours_in_progress"].notna() & resolved["issue_type"].isin(STORY_TYPES),
                                    "resolved_at"].to_numpy())
    for moment in differ["snapshot_time"].to_numpy(dtype="datetime64[us]"):
        n = np.searchsorted(resolved, moment, side="right")
        assert n > 50 and resolved[n - 51] == resolved[n - 50]
