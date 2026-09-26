"""Sprint updates from the platform: one sprint at a time, as it changes, with the outcomes at its close."""

from datetime import UTC, datetime, timedelta

from app import history

NOW = datetime.now(UTC).replace(microsecond=0)
DAY = timedelta(days=1)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _sprint(started: datetime, stories: list[dict], closed: datetime | None = None, name: str = "Sprint") -> dict:
    return {"name": name, "started_at": _iso(started), "planned_end": _iso(started + 14 * DAY),
            "closed_at": _iso(closed) if closed else None, "stories": stories}


def _story(story_id: str, committed: datetime, points: float = 3, **fields) -> dict:
    return {"story_id": story_id, "committed_at": _iso(committed), "points_at_commit": points, **fields}


def _estimate(client, *ids: str, sprint_id: str) -> dict[str, str]:
    stories = [{"story_id": i, "title": f"As a user I want feature {i} so that I can use it", "story_points": 3}
               for i in ids]
    predictions = client.post("/api/v1/estimate", json={"project_id": "TEAM", "sprint_id": sprint_id,
                                                        "stories": stories}).json()["predictions"]
    return {p["story_id"]: p["prediction_id"] for p in predictions}


def test_a_sprint_sent_as_it_starts_and_again_as_it_closes(client):
    started = NOW - 13 * DAY
    running = [_story("A", started), _story("B", started, points=5)]
    answer = client.put("/api/v1/projects/TEAM/sprints/S1", json=_sprint(started, running))
    assert answer.status_code == 200, answer.text
    assert answer.json() == {"project_id": "TEAM", "sprint_id": "S1", "closed": False, "stories": 2,
                             "outcomes_recorded": 0}
    shown = client.get("/api/v1/projects/TEAM/history").json()["sprints"]
    assert [(s["sprint_id"], s["closed_at"], s["stories"]) for s in shown] == [("S1", None, 2)]

    predicted = _estimate(client, "A", "B", sprint_id="S1")
    closing = NOW + timedelta(minutes=5)  # after the predictions were made
    done = [_story("A", started, points_at_close=3, done_in_sprint=True),
            _story("B", started, points=5, points_at_close=5, done_in_sprint=False)]
    answer = client.put("/api/v1/projects/TEAM/sprints/S1", json=_sprint(started, done, closing)).json()
    assert (answer["closed"], answer["outcomes_recorded"]) == (True, 2)

    sprint = client.get("/api/v1/projects/TEAM/history").json()["sprints"][0]
    assert (sprint["completed_points"], sprint["spilled_over"]) == (3, 1)  # B's R1 derived: its first sprint
    outcome = client.get(f"/api/v1/predictions/{predicted['B']}").json()["outcomes"]
    assert [(o["completed_in_sprint"], o["actual_story_points"]) for o in outcome] == [(False, 5.0)]

    # Sent again (e.g. B turned out reopened-free and A was re-pointed): outcomes are updated, not duplicated.
    done[0]["points_at_close"] = 5
    client.put("/api/v1/projects/TEAM/sprints/S1", json=_sprint(started, done, closing))
    outcomes = client.get(f"/api/v1/predictions/{predicted['A']}").json()["outcomes"]
    assert [(o["completed_in_sprint"], o["actual_story_points"]) for o in outcomes] == [(True, 5.0)]


def test_spillover_is_derived_only_at_a_storys_first_sprint(client):
    first, second = NOW - 40 * DAY, NOW - 26 * DAY
    client.put("/api/v1/projects/TEAM/sprints/S1", json=_sprint(first, [
        _story("B", first, points_at_close=5, done_in_sprint=False)], first + 14 * DAY))
    client.put("/api/v1/projects/TEAM/sprints/S2", json=_sprint(second, [
        _story("B", second, points_at_close=5, done_in_sprint=True),  # carried over, finished here
        _story("C", second, points_at_close=2, done_in_sprint=False)], second + 14 * DAY))
    _, items = history.SOURCE.records("TEAM")
    spilled = {(row.sprint_id, row.story_id): row.spilled_over for row in items.itertuples()}
    assert spilled == {("S1", "B"): True, ("S2", "B"): None, ("S2", "C"): True}


def test_a_sprint_with_problems_changes_nothing(client):
    started = NOW - 5 * DAY
    bad = _sprint(started, [_story("A", started, left_at=_iso(started - DAY)), _story("A", started)])
    bad["planned_end"] = _iso(started - DAY)
    answer = client.put("/api/v1/projects/TEAM/sprints/S1", json=bad)
    assert answer.status_code == 422
    fields = {p["column"] for p in answer.json()["detail"]["problems"]}
    assert fields == {"planned_end", "stories[0].left_at", "stories[1].story_id"}
    assert client.get("/api/v1/projects/TEAM/history").json()["sprints"] == []
    assert client.put("/api/v1/projects/TEAM/sprints/S1", json={"stories": []}).status_code == 422  # no dates


def test_removing_a_sprint(client):
    started = NOW - 5 * DAY
    client.put("/api/v1/projects/TEAM/sprints/S1", json=_sprint(started, [_story("A", started)]))
    assert client.delete("/api/v1/projects/TEAM/sprints/S1").json()["removed"] is True
    assert client.delete("/api/v1/projects/TEAM/sprints/S1").json()["removed"] is False
    assert client.get("/api/v1/projects/TEAM/history").json()["sprints"] == []


def test_three_closed_sprints_end_the_cold_start(client):
    for k in range(3):
        started = NOW - (60 - 14 * k) * DAY
        client.put(f"/api/v1/projects/TEAM/sprints/S{k + 1}", json=_sprint(started, [
            _story(f"S{k}-{i}", started, points_at_close=3, done_in_sprint=i < 4) for i in range(5)],
            started + 14 * DAY))
    shown = client.get("/api/v1/projects/TEAM/history").json()
    assert (shown["closed_sprints"], shown["cold_start"], shown["sources"]) == (3, False, ["platform"])
    assert shown["team_context"]["velocity_mean"] == 12  # 4 stories of 3 points done each sprint
    sources = client.post("/api/v1/estimate", json={"project_id": "TEAM", "stories": [
        {"story_id": "N", "title": "As a user I want reports so that I can share them"}]}).json()
    assert sources["predictions"][0]["feature_sources"]["historical_sprint"] == "history"
