import pandas as pd

from erp.labels import review, rules

T = pd.Timestamp


def test_first_event_after_is_strictly_after_the_moment():
    events = pd.DataFrame({"Issue_ID": [1, 1, 1, 2], "Creation_Date": [T("2020-01-01"), T("2020-01-03"),
                                                                      T("2020-01-05"), T("2020-01-01")]})
    after = pd.Series({1: T("2020-01-03"), 2: T("2020-01-02"), 3: T("2020-01-01")})
    assert rules.first_event_after(events, after).to_dict() == {1: T("2020-01-05"), 2: pd.NaT, 3: pd.NaT}


def test_blocking_links_pairs_additions_with_removals_and_ignores_other_link_types():
    changes = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Issue_ID": [10, 10, 10, 10],
        "Creation_Date": [T("2020-01-01"), T("2020-01-02"), T("2020-01-04"), T("2020-01-05")],
        "From_String": [None, None, "This issue is blocked by ABC-1", None],
        "To_String": ["This issue is blocked by ABC-1", "This issue relates to ABC-9", None,
                      "This issue depends on XYZ-7"],
    })
    links = rules.blocking_links(changes, pd.Series({"ABC-1": 101}))
    rows = sorted((r.blocker_key, r.added_at, r.removed_at, r.Blocker_ID) for r in links.itertuples())
    assert rows[0] == ("ABC-1", T("2020-01-01"), T("2020-01-04"), 101)
    assert rows[1][:2] == ("XYZ-7", T("2020-01-05")) and pd.isna(rows[1][2]) and pd.isna(rows[1][3])


def test_open_periods_follow_resolutions_and_reopenings():
    changes = pd.DataFrame({
        "ID": [1, 2, 3], "Issue_ID": [5, 5, 5],
        "Creation_Date": [T("2020-01-03"), T("2020-01-04"), T("2020-01-06")],
        "To_String": ["Fixed", None, "Fixed"],
    })
    created = pd.Series({5: T("2020-01-01"), 6: T("2020-01-02")})
    periods = rules.open_periods(changes, created)
    rows = sorted((r.Issue_ID, r.opened_at, r.closed_at) for r in periods.itertuples())
    assert rows[:2] == [(5, T("2020-01-01"), T("2020-01-03")), (5, T("2020-01-04"), T("2020-01-06"))]
    assert rows[2][:2] == (6, T("2020-01-02")) and pd.isna(rows[2][2])  # never resolved


def test_covered_time_counts_overlaps_once():
    intervals = [(T("2020-01-01"), T("2020-01-03")), (T("2020-01-02"), T("2020-01-04")),
                 (T("2020-01-10"), T("2020-01-11")), (T("2020-01-12"), T("2020-01-12"))]
    assert rules.covered_time(intervals) == pd.Timedelta(days=4)
    assert rules.covered_time([]) == pd.Timedelta(0)


def test_blocked_time_needs_the_link_and_an_open_blocker_inside_the_window():
    links = pd.DataFrame({
        "Issue_ID": [1, 1, 2], "blocker_key": ["A-1", "A-2", "A-1"],
        "added_at": [T("2020-01-02"), T("2020-01-01"), T("2020-01-01")],
        "removed_at": [pd.NaT, T("2020-01-03"), pd.NaT],
        "Blocker_ID": pd.array([101, 102, 101], dtype="Int64"),
    })
    blocker_open = pd.DataFrame({"Issue_ID": [101, 102], "opened_at": [T("2019-12-01"), T("2019-12-01")],
                                 "closed_at": [T("2020-01-06"), pd.NaT]})
    windows = pd.DataFrame({"start": [T("2020-01-01"), T("2020-01-07")], "end": [T("2020-01-15"), T("2020-01-15")]},
                           index=[1, 2])
    blocked = rules.blocked_time(links, blocker_open, windows)
    # story 1: A-2 blocks 01-01..01-03, A-1 blocks 01-02..01-06 (then resolved): 5 days together
    # story 2: its only blocker was resolved before the story's window started
    assert blocked.to_dict() == {1: pd.Timedelta(days=5), 2: pd.Timedelta(0)}


def test_risk_level_counts_rules():
    assert rules.risk_level(pd.Series([0, 1, 2, 5])).tolist() == ["low", "medium", "high", "high"]


def labels_table(n=400):
    frame = pd.DataFrame({"Issue_ID": range(n)})
    for i, rule in enumerate(["r1", "r2", "r3", "r4", "r5", "r6"]):
        frame[rule] = (frame["Issue_ID"] % 7 == i) & (frame["Issue_ID"] < 300)
    frame["at_risk"] = frame[["r1", "r2", "r3", "r4", "r5", "r6"]].any(axis=1)
    return frame


def test_review_sample_is_balanced_covers_rare_rules_and_repeats():
    labels = labels_table()
    sample = review.sample_for_review(labels, n=100, per_rule=5)
    assert len(sample) == 100 and sample["Issue_ID"].is_unique
    assert sample["at_risk"].sum() == 50
    for rule in review.RARE_RULES:
        assert sample[rule].sum() >= 5
    assert (sample["stratum"] == "at risk (rare rule)").sum() == 20
    again = review.sample_for_review(labels, n=100, per_rule=5)
    assert again["Issue_ID"].tolist() == sample["Issue_ID"].tolist()


def test_browse_url_turns_api_links_into_pages():
    api = "https://issues.apache.org/jira/rest/api/2/issue/13244150"
    assert review.browse_url(api, "MESOS-9887") == "https://issues.apache.org/jira/browse/MESOS-9887"
    assert review.browse_url(None, "X-1") is None


def test_history_text_keeps_the_sprint_window_and_skips_noise():
    story = pd.Series({"sprint_start": T("2020-01-10"), "sprint_closed": T("2020-01-24"),
                       "commitment_time": T("2020-01-10"), "created": T("2019-12-01"), "Sprint_Name": "S1"})
    events = pd.DataFrame({
        "Creation_Date": [T("2019-12-02"), T("2020-01-12"), T("2020-01-13"), T("2020-01-14"), T("2020-01-15")],
        "Field": ["status", "status", "status", "Link", "Story Points"],
        "From_String": ["Open", "Open", "In Progress", None, "3"],
        "To_String": ["Open", "In Progress", "In Progress", "This issue relates to A-1", "5"],
    })
    lines = review.history_text(events, story).split("\n")
    assert lines == ["2020-01-10 00:00  ▶ First sprint starts: S1", "2020-01-12 00:00  Status: Open → In Progress",
                     "2020-01-15 00:00  Story points: 3 → 5", "2020-01-24 00:00  ■ First sprint closed: S1"]
