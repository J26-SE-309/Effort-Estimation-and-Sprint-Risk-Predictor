"""Reading back what the service recorded: predictions, one prediction in full, and a project's summary."""

from app import store


def _stories(*ids: str) -> list[dict]:
    return [{"story_id": i, "title": f"As a user I want feature {i} so that I can use it", "story_points": 3}
            for i in ids]


def _estimate(client, stories: list[dict], sprint_id: str = "S1", project: str = "TEAM") -> list[dict]:
    body = {"project_id": project, "sprint_id": sprint_id, "stories": stories}
    return client.post("/api/v1/estimate", json=body).json()["predictions"]


def test_predictions_are_listed_newest_first_with_their_decision_and_outcome(client):
    first = _estimate(client, _stories("A", "B"))
    _estimate(client, _stories("C"), sprint_id="S2")
    a = first[0]["prediction_id"]
    client.post("/api/v1/feedback", json={"prediction_id": a, "decision": "adjust", "adjusted_story_points": 5})
    client.post("/api/v1/outcomes", json={"prediction_id": a, "completed_in_sprint": False,
                                          "actual_story_points": 5})

    listed = client.get("/api/v1/projects/TEAM/predictions").json()
    stories = [p["story_id"] for p in listed["predictions"]]
    assert stories[0] == "C" and sorted(stories) == ["A", "B", "C"] and listed["next_offset"] is None
    shown = next(p for p in listed["predictions"] if p["story_id"] == "A")
    assert shown["feedback"] == "adjust" and shown["outcome"] == {"completed_in_sprint": False,
                                                                  "actual_story_points": 5.0, "reopened": False}
    assert next(p for p in listed["predictions"] if p["story_id"] == "B")["outcome"] is None

    page = client.get("/api/v1/projects/TEAM/predictions", params={"limit": 2}).json()
    assert len(page["predictions"]) == 2 and page["next_offset"] == 2
    rest = client.get("/api/v1/projects/TEAM/predictions", params={"limit": 2, "offset": 2}).json()
    assert len(rest["predictions"]) == 1 and rest["next_offset"] is None
    by_sprint = client.get("/api/v1/projects/TEAM/predictions", params={"sprint_id": "S2"}).json()
    assert [p["story_id"] for p in by_sprint["predictions"]] == ["C"]
    by_story = client.get("/api/v1/projects/TEAM/predictions", params={"story_id": "B"}).json()
    assert [p["story_id"] for p in by_story["predictions"]] == ["B"]
    assert client.get("/api/v1/projects/OTHER/predictions").json()["predictions"] == []
    assert client.get("/api/v1/projects/TEAM/predictions", params={"limit": 500}).status_code == 422


def test_one_prediction_in_full(client):
    prediction = _estimate(client, _stories("A"))[0]
    client.post("/api/v1/feedback", json={"prediction_id": prediction["prediction_id"], "decision": "accept"})
    detail = client.get(f"/api/v1/predictions/{prediction['prediction_id']}").json()
    assert detail["prediction"]["predicted_story_points"] == prediction["predicted_story_points"]
    assert detail["features"]["story_points"] == 3 and "team_velocity_rolling" in detail["features"]  # FR21
    assert [f["decision"] for f in detail["feedback"]] == ["accept"] and detail["outcomes"] == []
    assert client.get("/api/v1/predictions/no-such-id").status_code == 404


def test_the_project_summary(client):
    first = _estimate(client, _stories("A", "B"))
    _estimate(client, _stories("C"), sprint_id="S2")
    a = first[0]
    client.post("/api/v1/feedback", json={"prediction_id": a["prediction_id"], "decision": "reject"})
    client.post("/api/v1/outcomes", json={"prediction_id": a["prediction_id"], "completed_in_sprint": True,
                                          "actual_story_points": 8})
    summary = client.get("/api/v1/projects/TEAM/summary").json()
    assert (summary["predictions"], summary["stories"], summary["pinned"]) == (3, 3, 0)
    assert summary["by_configuration"] == {"fasttext-lightgbm": 3} and sum(summary["by_risk_level"].values()) == 3
    assert summary["feedback"] == {"reject": 1} and summary["outcomes"] == 1
    assert summary["completed_share"] == 1.0
    assert abs(summary["effort_mae"] - abs(a["predicted_story_points"] - 8)) < 1e-9
    assert abs(summary["mean_spillover_probability"] - a["spillover_probability"]) < 1e-9
    one_sprint = client.get("/api/v1/projects/TEAM/summary", params={"sprint_id": "S2"}).json()
    assert (one_sprint["predictions"], one_sprint["outcomes"], one_sprint["effort_mae"]) == (1, 0, None)


def test_reading_back_without_a_database_answers_503(client, monkeypatch):
    prediction = _estimate(client, _stories("A"))[0]
    monkeypatch.setattr(store, "_usable", lambda: False)
    for path in ("/api/v1/projects/TEAM/predictions", f"/api/v1/predictions/{prediction['prediction_id']}",
                 "/api/v1/projects/TEAM/summary"):
        assert client.get(path).status_code == 503
