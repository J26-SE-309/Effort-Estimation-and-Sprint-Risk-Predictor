import numpy as np
import pandas as pd

from erp.features import catalog, team, text

T = pd.Timestamp


def test_vague_terms_and_ambiguity_score():
    description = "The page should load fast. Show the report. Maybe add some filters, etc."
    assert text.vague_terms(description) == ["fast", "maybe", "some", "etc"]
    assert text.ambiguity_score(description) == 2 / 3
    assert text.vague_terms("breakfast is handled elsewhere") == []  # whole words only
    assert text.ambiguity_score("") == 0.0


def test_acceptance_criteria_detection_and_count():
    ac = "Build the page.\nAcceptance criteria:\n* shows the list\n* can sort\n* can filter\n* exports CSV"
    assert text.acceptance_criteria(ac) == (True, 4)
    gwt = "Given a logged-in user, when they open the page, then the list shows. Given no data, when ..., then ..."
    assert text.acceptance_criteria(gwt) == (True, 2)
    assert text.acceptance_criteria("Just do it") == (False, 0)


def test_missing_info_flags():
    assert text.missing_info_flags("Crash", "It crashes", "Bug") == 2  # short, and no steps to reproduce
    steps = "Steps to reproduce: open the admin page, click save twice and wait for the dialog to appear."
    assert text.missing_info_flags("Crash on save", steps, "Bug") == 0
    story = "As a reviewer I want to filter the queue so that I find my items faster than today in the tool."
    assert text.missing_info_flags("Filter queue", story, "Story") == 0
    assert text.missing_info_flags("Filter queue", "TBD", "Story") == 3  # short, placeholder, no user or goal


def test_is_started():
    status = pd.Series(["Open", "In Progress", "Waiting for peer review", "In Progress", None])
    resolved = pd.Series([False, False, False, True, False])
    assert team.is_started(status, resolved).tolist() == [False, True, True, False, False]


def test_previous_sprint_stats_only_uses_closed_sprints():
    sprints = pd.DataFrame({"Project_ID": 1, "closed_at": [T("2020-01-10"), T("2020-01-20"), T("2020-01-30"),
                                                            T("2020-02-09")],
                            "completed_points": [10.0, 20.0, 30.0, 40.0]})
    stories = pd.DataFrame({"Project_ID": [1, 1, 1], "snapshot_time": [T("2020-01-05"), T("2020-01-20"),
                                                                       T("2020-02-01")]}, index=[7, 8, 9])
    out = team.previous_sprint_stats(sprints, stories)
    assert np.isnan(out.loc[7, "team_velocity_rolling"]) and out.loc[7, "history_sprints"] == 0
    assert out.loc[8, "team_velocity_rolling"] == 15.0 and out.loc[8, "velocity_variance"] == 50.0
    assert out.loc[9, "team_velocity_rolling"] == 20.0 and out.loc[9, "history_sprints"] == 3


def test_recent_rate_needs_enough_known_outcomes():
    events = pd.DataFrame({"Project_ID": 1, "r1": [1, 0, 1, 1, 0, 1],
                           "known": pd.date_range("2020-01-01", periods=6, freq="D")})
    stories = pd.DataFrame({"Project_ID": [1, 1, 2], "snapshot_time": [T("2020-01-04"), T("2020-01-10"),
                                                                       T("2020-01-10")]}, index=[1, 2, 3])
    out = team.recent_rate(events, stories, "r1", "known", window=5, minimum=5)
    assert np.isnan(out[1]) and np.isnan(out[3])  # 4 known outcomes; another project
    assert out[2] == 3 / 5  # the last five of six


def test_catalog_names_are_unique_and_grouped():
    names = catalog.names()
    assert len(names) == len(set(names))
    assert set(catalog.UPSTREAM_GROUPS) <= set(catalog.GROUPS)
    assert "ambiguity_score" in catalog.names("requirement_quality")
