"""Sprint history: the CSV import and its checks, the history view, estimates using it, and development data."""

import time

import pandas as pd
import pytest
from sqlalchemy import select

from app import db, devdata, history, store
from app.tables import PredictionRecord

TODAY = pd.Timestamp("2026-09-26T12:00:00Z")
CSV = {"Content-Type": "text/csv"}


def _steady_csv() -> str:
    return history.to_csv(*devdata.synthetic("SYN-STEADY", devdata.SYNTHETIC["SYN-STEADY"], TODAY))


def _estimate(client, project_id: str, sprint_id: str | None = None, **context) -> dict:
    stories = [{"story_id": "NEW-1", "title": "As a user I want to export reports so that I can share them",
                "story_points": 3}]
    return client.post("/api/v1/estimate", json={"project_id": project_id, "sprint_id": sprint_id,
                                                 "stories": stories, **context}).json()


def _snapshot(prediction_id: str) -> dict:
    with db.SessionLocal() as session:
        return session.scalar(select(PredictionRecord.features).where(PredictionRecord.id == prediction_id))


def test_import_then_the_history_the_models_see(client):
    imported = client.post("/api/v1/projects/TEAM/history", content=_steady_csv(), headers=CSV)
    assert imported.status_code == 200, imported.text
    assert imported.json()["sprints"] == 9 and imported.json()["source"] == "imported"

    body = client.get("/api/v1/projects/TEAM/history").json()
    assert (body["closed_sprints"], body["cold_start"], body["sprints_needed"]) == (8, False, 0)
    assert body["sources"] == ["imported"]
    newest = body["sprints"][0]
    assert newest["closed_at"] is None and newest["completed_points"] is None  # the running sprint
    last_three = [s["completed_points"] for s in body["sprints"][1:4]]
    assert body["team_context"]["velocity_mean"] == pytest.approx(sum(last_three) / 3)
    assert 0 <= body["team_context"]["spillover_rate"] <= 1

    unknown = client.get("/api/v1/projects/NOBODY/history").json()
    assert (unknown["closed_sprints"], unknown["cold_start"], unknown["sprints"]) == (0, True, [])


def test_import_lists_every_problem_and_imports_nothing(client):
    csv = "\n".join([
        "sprint_id,sprint_started_at,sprint_planned_end,story_id,committed_at,points_at_commit,done_in_sprint,colour",
        "S1,2026-08-03T09:00:00Z,2026-08-17T09:00:00Z,A,yesterday,3,true,red",
        "S1,2026-08-03T09:00:00Z,2026-08-17T09:00:00Z,B,2026-08-03T09:00:00Z,-2,maybe,red",
        "S1,2026-08-04T09:00:00Z,2026-08-17T09:00:00Z,B,2026-08-03T09:00:00Z,1,true,red",
        ",,,C,2026-08-03T09:00:00Z,,,red",
    ])
    response = client.post("/api/v1/projects/TEAM/history", content=csv, headers=CSV)
    assert response.status_code == 422
    found = {(p["row"], p["column"]) for p in response.json()["detail"]["problems"]}
    assert found == {(None, "colour"), (2, "committed_at"), (3, "points_at_commit"), (3, "done_in_sprint"),
                     (4, "story_id"), (2, "sprint_started_at"), (5, "committed_at")}
    assert client.get("/api/v1/projects/TEAM/history").json()["sprints"] == []
    assert client.post("/api/v1/projects/TEAM/history", content=csv).status_code == 415  # not sent as text/csv


def test_estimates_use_the_stored_history(client):
    cold = _estimate(client, "TEAM")["predictions"][0]
    assert cold["feature_sources"]["historical_sprint"] == "missing"

    client.post("/api/v1/projects/TEAM/history", content=_steady_csv(), headers=CSV)
    history.forget()  # another worker process may hold the empty history until it re-reads (REFRESH_SECONDS)
    warm = _estimate(client, "TEAM", sprint_id="SYN-STEADY-S9")["predictions"][0]
    sources = warm["feature_sources"]
    assert sources["historical_sprint"] == "history" and sources["sprint_context"] == "history"
    team = client.get("/api/v1/projects/TEAM/history").json()["team_context"]
    seen = _snapshot(warm["prediction_id"])
    assert seen["history_sprints"] == 8 and seen["team_velocity_rolling"] == pytest.approx(team["velocity_mean"])
    assert seen["sprint_length_days"] == 14
    assert warm["confidence_score"] > cold["confidence_score"]  # no longer a cold start


def test_the_callers_own_team_context_wins(client):
    client.post("/api/v1/projects/TEAM/history", content=_steady_csv(), headers=CSV)
    prediction = _estimate(client, "TEAM", team_context={"velocity_mean": 99})["predictions"][0]
    assert prediction["feature_sources"]["historical_sprint"] == "request"
    seen = _snapshot(prediction["prediction_id"])
    assert seen["team_velocity_rolling"] == 99 and seen["history_sprints"] == 8  # the rest from the history


def test_an_unreadable_history_is_a_cold_start(client, monkeypatch):
    client.post("/api/v1/projects/TEAM/history", content=_steady_csv(), headers=CSV)
    history.forget()
    monkeypatch.setattr(store, "_usable", lambda: False)
    prediction = _estimate(client, "TEAM")["predictions"][0]
    assert prediction["feature_sources"]["historical_sprint"] == "missing"


@pytest.mark.parametrize("name", list(devdata.SYNTHETIC))
def test_synthetic_teams_pass_the_import_and_are_what_they_claim(name):
    team = devdata.SYNTHETIC[name]
    sprints, items, problems = history.parse_csv(history.to_csv(*devdata.synthetic(name, team, TODAY)))
    assert problems == []
    assert sprints["closed_at"].notna().sum() == team.closed and sprints["closed_at"].isna().sum() == 1
    first, again = devdata.synthetic(name, team, TODAY)
    assert first.equals(devdata.synthetic(name, team, TODAY)[0]) and again.equals(
        devdata.synthetic(name, team, TODAY)[1])  # seeded: the same team every time


def test_development_data_is_removed_with_its_predictions(client):
    devdata.load_synthetic(TODAY)
    assert set(history.projects()) == set(devdata.SYNTHETIC)
    dev = _estimate(client, "SYN-STEADY")["predictions"][0]["prediction_id"]
    real = _estimate(client, "REAL")["predictions"][0]["prediction_id"]
    client.put("/api/v1/projects/SYN-STEADY/pin", json={"configuration_id": "tfidf-rf"})

    assert devdata.remove(devdata.DEVELOPMENT) == sorted(devdata.SYNTHETIC)
    assert history.projects() == {}
    assert _snapshot(dev) is None and _snapshot(real) is not None
    store.reset()
    assert client.get("/api/v1/projects/SYN-STEADY/pin").json()["configuration_id"] is None


def test_an_old_copy_answers_at_once_while_the_records_are_read_again(monkeypatch):
    reads = []

    class Source:
        def records(self, project_id):
            reads.append(project_id)
            return devdata.synthetic("SYN-STEADY", devdata.SYNTHETIC["SYN-STEADY"], TODAY)

    monkeypatch.setattr(history, "SOURCE", Source())
    history.forget()
    store.reset()
    first = history.records("P")
    assert reads == ["P"] and history.records("P") is first  # a fresh copy: nothing read again
    monkeypatch.setattr(history, "REFRESH_SECONDS", -1)
    assert history.records("P") is first  # an old copy still answers at once...
    for _ in range(200):  # ...while a background thread reads the records again
        if history._copies["P"] is not first:
            break
        time.sleep(0.01)
    assert reads == ["P", "P"] and history._copies["P"] is not first
    history.forget()
