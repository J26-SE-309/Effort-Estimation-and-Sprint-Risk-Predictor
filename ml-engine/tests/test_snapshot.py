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


def test_value_at_replays_the_field_backwards():
    changes = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Issue_ID": [10, 10, 10, 20],
        "Creation_Date": [T("2020-01-01"), T("2020-01-05"), T("2020-01-09"), T("2020-01-03")],
        "From_String": [None, "3", "5", "2"],  # issue 10: None -> 3 -> 5 -> 8; issue 20: 2 -> 1
    })
    at = pd.Series({10: T("2020-01-05"), 20: T("2020-01-04"), 30: T("2020-01-04")})
    current = pd.Series({10: 8, 20: 1, 30: 13})
    out = history.value_at(changes, at, current)
    assert out["value"].to_dict() == {10: "5", 20: 1, 30: 13}  # the change at exactly 01-05 is already made
    assert out["changed_later"].to_dict() == {10: True, 20: False, 30: False}


def test_value_at_before_any_estimate_is_missing():
    changes = pd.DataFrame({"ID": [1], "Issue_ID": [10], "Creation_Date": [T("2020-01-05")], "From_String": [None]})
    out = history.value_at(changes, pd.Series({10: T("2020-01-01")}), pd.Series({10: 3}))
    assert pd.isna(out.loc[10, "value"]) and out.loc[10, "changed_later"]


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
