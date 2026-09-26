import numpy as np
import pandas as pd
import pytest

from erp import tawos
from erp.snapshot import build_snapshot, filters, history, text, timeline

T = pd.Timestamp


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('"Fix stream failover "', "Fix stream failover "),
        ('"Say ""hello"" twice"', 'Say "hello" twice'),
        ("not quoted", "not quoted"),
        (None, None),
    ],
)
def test_unquote(value, expected):
    assert tawos.unquote(value) == expected


@pytest.mark.parametrize(
    ("name", "level"),
    [("Major - P3", "medium"), ("Medium", "medium"), ("Blocker", "highest"), ("Minor", "low"),
     ("To be reviewed", "unknown"), ("", "unknown"), (None, "unknown")],
)
def test_priority_level(name, level):
    assert tawos.priority_level(name) == level


def test_clean_text_removes_code_blocks_and_collapses_whitespace():
    raw = "Steps:\n\n{code:java}\nint x = 1;\n{code}\nthen   {noformat}log line{noformat} fails.\n{code}never closed"
    assert text.clean_text(raw) == "Steps: then fails."
    assert text.clean_text('"Quoted ""title"""', quoted=True) == 'Quoted "title"'
    assert text.clean_text(None) == ""
    assert text.has_code(raw) and not text.has_code("plain words") and not text.has_code(float("nan"))


def test_value_at_reads_the_last_change_before_the_moment():
    changes = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Issue_ID": [10, 10, 10, 20],
        "Creation_Date": [T("2020-01-01"), T("2020-01-05"), T("2020-01-09"), T("2020-01-03")],
        "From_String": [None, "3", "5", "2"],
        "To_String": ["3", "5", "8", "1"],  # issue 10: none -> 3 -> 5 -> 8; issue 20: 2 -> 1
    })
    at = pd.Series({10: T("2020-01-05"), 20: T("2020-01-04"), 30: T("2020-01-04")})
    current = pd.Series({10: "8", 20: "1", 30: "13"})
    out = history.value_at(changes, at, current)
    assert out["value"].to_dict() == {10: "5", 20: "1", 30: "13"}  # the change at exactly 01-05 is already made
    assert out["from_log"].to_dict() == {10: True, 20: True, 30: False}
    assert out["changed_later"].to_dict() == {10: True, 20: False, 30: False}


def test_value_at_before_the_first_change_uses_what_it_replaced():
    changes = pd.DataFrame({"ID": [1], "Issue_ID": [10], "Creation_Date": [T("2020-01-05")],
                            "From_String": ["2"], "To_String": ["3"]})
    out = history.value_at(changes, pd.Series({10: T("2020-01-01"), 11: T("2020-01-01")}), pd.Series({10: "3"}))
    assert out.loc[10, "value"] == "2" and out.loc[10, "changed_later"]  # set at creation, changed later
    assert pd.isna(out.loc[11, "value"]) and not out.loc[11, "from_log"]


def test_value_at_survives_a_missing_event():
    # Resolved in 2016, reopened in 2018 without a logged change, resolved again: the 2018 change claims the
    # field was empty before, but in 2016 the story was resolved.
    changes = pd.DataFrame({"ID": [1, 2], "Issue_ID": [7, 7], "Creation_Date": [T("2016-02-10"), T("2018-11-26")],
                            "From_String": [None, None], "To_String": ["Fixed", "Fixed"]})
    out = history.value_at(changes, pd.Series({7: T("2016-02-11")}), pd.Series({7: "Fixed"}))
    assert out.loc[7, "value"] == "Fixed"


def test_snapshot_time_is_start_plus_tolerance_for_planned_and_join_time_for_mid_sprint():
    frame = pd.DataFrame({
        "entry": ["planned", "added_mid_sprint"],
        "Start_Date": [T("2020-01-01 09:00"), T("2020-01-01 09:00")],
        "commitment_time": [T("2020-01-01 09:00"), T("2020-01-04 15:00")],
    })
    times = build_snapshot.snapshot_times(frame).tolist()
    assert times == [T("2020-01-01 09:00") + timeline.CLOCK_TOLERANCE, T("2020-01-04 15:00")]


def test_filter_log_records_each_step():
    log = filters.FilterLog(pd.DataFrame({"x": range(10)}), "All")
    log.keep(log.frame["x"] >= 3, "At least 3")
    log.keep(log.frame["x"] % 2 == 0, "Even")
    assert log.table().to_dict("list") == {
        "Step": ["All", "At least 3", "Even"], "Removed": ["", "−3", "−4"], "Remaining": ["10", "7", "3"]}


def test_projects_using_sprints():
    issues = pd.DataFrame({
        "Project_ID": [1] * 4 + [2] * 4,
        "Type": ["Story", "Bug", "Task", "Epic", "Story", "Story", "Story", "Story"],
        "Story_Point": [3, 5, None, 8, 1, 2, 3, 5],
    }, index=range(1, 9))
    commitments = pd.DataFrame({
        "commitment_status": ["ok", "ok", "ok", "ok"],
        "commitment_project": [1, 1, 1, 2],
        "Sprint_ID": [100, 101, 102, 200],
    }, index=[1, 2, 3, 5])
    out = filters.projects_using_sprints(issues, commitments)
    # project 1: both estimated non-epic issues committed, but only 3 sprints; project 2: 1 of 4 committed
    assert out.loc[1, "share"] == 1.0 and out.loc[2, "share"] == 0.25
    assert not out["uses_sprints"].any()


def test_values_at_matches_value_at_for_many_moments():
    rng = np.random.default_rng(3)
    rows = []
    for issue in range(1, 30):
        value = None
        for _ in range(rng.integers(0, 5)):
            new = str(rng.integers(1, 9)) if rng.random() > 0.2 else None
            rows.append({"ID": len(rows) + 1, "Issue_ID": issue, "From_String": value, "To_String": new,
                         "Creation_Date": T("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 60)))})
            value = new
    changes = pd.DataFrame(rows)
    current = pd.Series({i: str(i) for i in range(1, 30)})
    moments = [T("2020-01-01") + pd.Timedelta(days=d) for d in (0, 10, 25, 59, 70)]
    pairs = pd.DataFrame([(i, m) for i in range(1, 30) for m in moments], columns=["Issue_ID", "at"])
    many = history.values_at(changes, pairs, current)
    for m in moments:
        one = history.value_at(changes, pd.Series({i: m for i in range(1, 30)}), current)["value"]
        got = many[pairs["at"] == m].set_axis(range(1, 30))
        assert [None if pd.isna(v) else v for v in got] == [None if pd.isna(v) else v for v in one]
