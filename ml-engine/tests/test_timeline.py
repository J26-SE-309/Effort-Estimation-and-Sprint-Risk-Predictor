import pandas as pd
import pytest

from erp.snapshot import timeline

T = pd.Timestamp


def log(*changes):
    """Change_Log rows for the Sprint field: (issue, from, to, when)."""
    rows = [
        {"ID": i, "Issue_ID": issue, "From_Value": old, "To_Value": new, "Creation_Date": T(when)}
        for i, (issue, old, new, when) in enumerate(changes, start=1)
    ]
    return pd.DataFrame(rows)


def stays_of(frame):
    return sorted(
        (r.Issue_ID, r.JiraID, r.joined_at, None if pd.isna(r.left_at) else r.left_at, r.join_logged)
        for r in frame.itertuples()
    )


def test_replay_keep_mode_leaves_the_closed_sprint_in_the_field():
    stays = timeline.replay_sprint_field(
        log((1, None, "103", "2020-01-01"), (1, "103", "103, 106", "2020-01-15")), pd.Series(dtype="datetime64[ns]"))
    assert stays_of(stays) == [(1, 103, T("2020-01-01"), None, True), (1, 106, T("2020-01-15"), None, True)]


def test_replay_replace_mode_closes_the_old_sprint():
    stays = timeline.replay_sprint_field(
        log((2, None, "107", "2020-01-01"), (2, "107", "108", "2020-01-15")), pd.Series(dtype="datetime64[ns]"))
    assert stays_of(stays) == [(2, 107, T("2020-01-01"), T("2020-01-15"), True), (2, 108, T("2020-01-15"), None, True)]


def test_replay_sprint_set_at_creation_gets_the_creation_date():
    created = pd.Series({3: T("2019-12-20")})
    stays = timeline.replay_sprint_field(log((3, "5", "5, 6", "2020-01-10")), created)
    assert stays_of(stays) == [(3, 5, T("2019-12-20"), None, False), (3, 6, T("2020-01-10"), None, True)]


def test_replay_removed_and_readded_sprint_gives_two_stays_and_order_follows_time():
    # stored out of time order: joins on 01-01, leaves on 01-02, joins again on 01-03
    changes = log((4, "", "9", "2020-01-03"), (4, None, "9", "2020-01-01"), (4, "9", "", "2020-01-02"))
    stays = timeline.replay_sprint_field(changes, pd.Series(dtype="datetime64[ns]"))
    assert stays_of(stays) == [(4, 9, T("2020-01-01"), T("2020-01-02"), True), (4, 9, T("2020-01-03"), None, True)]


def test_resolve_prefers_own_project_then_same_repository_and_never_another_repository():
    projects = pd.DataFrame({"ID": [1, 2, 3], "Repository_ID": [10, 10, 20]})
    sprints = pd.DataFrame({
        "ID": [100, 101, 102, 103], "JiraID": [7, 7, 8, 9], "Project_ID": [2, 1, 3, 2],
        "Name": ["a", "b", "c", "d"], "State": "CLOSED", "Start_Date": T("2020-01-01"),
        "End_Date": T("2020-01-14"), "Complete_Date": T("2020-01-14"),
    })
    issues = pd.DataFrame({"ID": [1, 2], "Project_ID": [1, 3]})
    stays = pd.DataFrame({"Issue_ID": [1, 1, 1, 2], "JiraID": [7, 9, 8, 7], "joined_at": T("2020-01-01"),
                          "left_at": pd.NaT, "join_logged": True})
    out = timeline.resolve_sprints(stays, issues, sprints, projects)
    assert out["Sprint_ID"].tolist() == [101, 103, pd.NA, pd.NA]
    assert out["Sprint_Name"].tolist()[:2] == ["b", "d"]


SPRINT_A = {"Sprint_ID": 1, "JiraID": 11, "Sprint_Name": "A", "State": "CLOSED", "Start_Date": T("2020-01-01 09:00"),
            "End_Date": T("2020-01-15 09:00"), "Complete_Date": T("2020-01-15 09:00")}
SPRINT_B = {"Sprint_ID": 2, "JiraID": 12, "Sprint_Name": "B", "State": "CLOSED", "Start_Date": T("2020-01-15 10:00"),
            "End_Date": T("2020-01-29 10:00"), "Complete_Date": T("2020-01-29 10:00")}
UNKNOWN = {"Sprint_ID": pd.NA, "JiraID": 99, "Sprint_Name": None, "State": None, "Start_Date": pd.NaT,
           "End_Date": pd.NaT, "Complete_Date": pd.NaT}


def stay(issue, sprint, joined, left=None, **changes):
    return {"Issue_ID": issue, "Project_ID": 1, "Repository_ID": 10, **sprint, **changes,
            "joined_at": T(joined), "left_at": T(left) if left else pd.NaT, "join_logged": True}


def memberships(*rows):
    frame = pd.DataFrame(list(rows))
    frame["Sprint_ID"] = frame["Sprint_ID"].astype("Int64")
    return timeline.sprint_memberships(frame).set_index(["Issue_ID", "JiraID"])


@pytest.mark.parametrize(
    ("joined", "left", "entry", "exit_", "committed"),
    [
        ("2019-12-30", None, "planned", "stayed", True),
        ("2020-01-01 11:00", None, "planned", "stayed", True),  # two hours after the start: within tolerance
        ("2020-01-04", None, "added_mid_sprint", "stayed", True),
        ("2019-12-20", "2019-12-31", "left_before_start", None, False),
        ("2019-12-30", "2020-01-06", "planned", "left_mid_sprint", True),
        ("2019-12-30", "2020-01-15 10:00", "planned", "left_at_close", True),  # an hour after the close
        ("2019-12-30", "2020-02-10", "planned", "left_after_close", True),
        ("2020-02-01", None, "joined_after_close", None, False),
        ("2020-01-04 10:00", "2020-01-04 10:40", "brief_stay", None, False),  # in and out within the hour
        ("2019-12-30", "2020-01-02 05:00", "brief_stay", None, False),  # dropped 20 hours after the start
        ("2020-01-15 01:00", None, "joined_at_close", None, False),  # eight hours before the close
    ],
)
def test_membership_entry_and_exit(joined, left, entry, exit_, committed):
    row = memberships(stay(1, SPRINT_A, joined, left)).loc[(1, 11)]
    assert (row["entry"], None if pd.isna(row["exit"]) else row["exit"], row["committed"]) == (entry, exit_, committed)


def test_commitment_time_is_the_start_for_planned_and_the_join_for_mid_sprint():
    m = memberships(stay(1, SPRINT_A, "2019-12-30"), stay(2, SPRINT_A, "2020-01-04 15:00"))
    assert m.loc[(1, 11), "commitment_time"] == SPRINT_A["Start_Date"]
    assert m.loc[(2, 11), "commitment_time"] == T("2020-01-04 15:00")


def test_spillover_in_replace_mode_numbers_both_sprints():
    m = memberships(stay(1, SPRINT_A, "2019-12-30", "2020-01-15 09:05"), stay(1, SPRINT_B, "2020-01-15 09:05"))
    assert m.loc[(1, 11), ["exit", "commit_order"]].tolist() == ["left_at_close", 1]
    assert m.loc[(1, 12), ["entry", "commit_order"]].tolist() == ["planned", 2]


def test_sprints_without_usable_dates_are_never_committed():
    future = {**SPRINT_B, "Sprint_ID": 3, "JiraID": 13, "State": "FUTURE", "Start_Date": pd.NaT,
              "End_Date": pd.NaT, "Complete_Date": pd.NaT}
    open_sprint = {**SPRINT_B, "Sprint_ID": 4, "JiraID": 14, "State": "ACTIVE", "Complete_Date": pd.NaT}
    m = memberships(stay(1, UNKNOWN, "2020-01-02"), stay(1, future, "2020-01-02"), stay(1, open_sprint, "2020-01-02"))
    assert m.loc[(1, 99), ["entry", "committed"]].tolist() == ["sprint_unknown", False]
    assert m.loc[(1, 13), ["entry", "committed"]].tolist() == ["sprint_not_started", False]
    assert m.loc[(1, 14), ["entry", "exit"]].tolist() == ["planned", "sprint_open"]


def first(*rows):
    frame = pd.DataFrame(list(rows))
    frame["Sprint_ID"] = frame["Sprint_ID"].astype("Int64")
    return timeline.first_commitments(timeline.sprint_memberships(frame))


def test_first_commitment_status():
    open_sprint = {**SPRINT_B, "Sprint_ID": 4, "JiraID": 14, "State": "ACTIVE", "Complete_Date": pd.NaT}
    out = first(
        stay(1, SPRINT_A, "2019-12-30", "2020-01-15 09:05"), stay(1, SPRINT_B, "2020-01-15 09:05"),
        stay(2, UNKNOWN, "2019-12-01", "2019-12-15"), stay(2, SPRINT_A, "2019-12-30"),  # unknown sprint came first
        stay(3, SPRINT_A, "2019-12-30"), stay(3, UNKNOWN, "2020-02-01"),  # unknown sprint came later: fine
        stay(4, SPRINT_A, "2019-12-20", "2019-12-31"),
        stay(5, UNKNOWN, "2020-01-02"),
        stay(6, open_sprint, "2020-01-14"),
    )
    assert out["status"].to_dict() == {1: "ok", 2: "earlier_sprint_unknown", 3: "ok", 4: "never_committed",
                                       5: "sprint_unknown", 6: "sprint_still_open"}
    assert out.loc[1, ["Sprint_Name", "n_sprints_committed"]].tolist() == ["A", 2]
    assert out.loc[4, "Project_ID"] == 1
