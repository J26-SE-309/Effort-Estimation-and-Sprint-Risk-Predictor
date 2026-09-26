"""The synthetic backlogs (app.backlogs): requests the service answers as they are, signals that follow the
stories, and committed files that match the generator."""

import json

import pandas as pd
import pytest

from app import backlogs, devdata

FILES = sorted(backlogs.all_backlogs())


def _file(name: str) -> dict:
    return json.loads((backlogs.OUT / name).read_text(encoding="utf-8"))


def _sources(client, backlog: dict) -> dict:
    return client.post("/api/v1/estimate", json=backlog).json()["predictions"][0]["feature_sources"]


def test_the_committed_files_are_what_the_generator_makes(tmp_path):
    for path in backlogs.write(tmp_path):  # compared as JSON: a checkout may change the line endings
        committed = backlogs.OUT / path.name
        assert committed.exists() and _file(path.name) == json.loads(path.read_text(encoding="utf-8")), \
            f"{path.name} is out of date: run python -m app.devdata backlogs"


@pytest.mark.parametrize("name", FILES)
def test_every_backlog_is_answered_as_it_is(client, name):
    backlog = _file(name)  # exactly what a caller would send, synthetic label included
    response = client.post("/api/v1/estimate", json=backlog)
    assert response.status_code == 200, response.text
    assert [p["story_id"] for p in response.json()["predictions"]] == [s["story_id"] for s in backlog["stories"]]


def test_the_components_signals_are_used_when_sent_and_replaced_when_not(client):
    full = _sources(client, _file("syn-steady.json"))
    assert (full["requirement_quality"], full["acceptance_criteria"], full["traceability"]) == \
        ("component", "component", "component")
    none = _sources(client, _file("syn-steady-without-components.json"))
    assert (none["requirement_quality"], none["acceptance_criteria"], none["traceability"]) == \
        ("proxy", "proxy", "missing")  # FR17: the service's own stand-ins, flagged
    one = _sources(client, _file("syn-steady-component-1-only.json"))
    assert (one["requirement_quality"], one["acceptance_criteria"]) == ("component", "proxy")


def test_a_team_backlog_uses_the_teams_stored_history(client):
    devdata.load_synthetic(pd.Timestamp("2026-09-26T12:00:00Z"))
    sources = _sources(client, _file("syn-steady.json"))  # for the team's running sprint
    assert (sources["historical_sprint"], sources["sprint_context"]) == ("history", "history")


def test_the_signals_follow_each_story():
    stories = [story for name in backlogs.TEAMS for story in backlogs.team_backlog(name)["stories"]]
    vague = [s["upstream"]["ambiguity_score"] for s in stories if s["issue_type"] == "Improvement"]
    clear = [s["upstream"]["ambiguity_score"] for s in stories if s["issue_type"] == "Story" and
             s["upstream"]["ac_completeness_score"] >= 0.8]
    assert vague and clear and min(vague) > max(clear)
    for story in stories:
        signals = story["upstream"]
        traces = [story["has_epic"], story["linked_issue_count"] > 0, signals["has_linked_tests"]]
        assert signals["traceability_coverage_pct"] == round(sum(traces) / 3, 2)
        assert signals["unlinked_artifact_count"] == 3 - sum(traces)
        assert signals["invest_compliance_flags"]["independent"] == (story["blocker_count"] == 0)
        assert (signals["ac_completeness_score"] > 0) == bool(story["acceptance_criteria"])
    erratic = backlogs.team_backlog("SYN-ERRATIC")["stories"]
    assert len({s["title"] for s in erratic}) == len(erratic)  # no story twice in one backlog


def test_sprints_are_planned_against_the_teams_velocity():
    """Only the erratic team over-commits; the other sprints fit, so only its stories are told to cut scope."""
    for name, (load, _, _) in backlogs.TEAMS.items():
        stories = backlogs.team_backlog(name)["stories"]
        committed = sum(s["story_points"] or 0 for s in stories)
        if load is None:
            assert committed == 0 and len(stories) == backlogs.UNESTIMATED_STORIES
        elif load <= 1:
            assert 0.75 * load * backlogs.capacity(name) <= committed <= load * backlogs.capacity(name)
        else:
            assert committed >= load * backlogs.capacity(name)


def test_requests_outside_the_contract_are_refused(client):
    story = backlogs.edge_cases()["stories"][0]

    def status(stories: list[dict]) -> int:
        return client.post("/api/v1/estimate", json={"project_id": "SYN-EDGE", "stories": stories}).status_code

    for bad in ({**story, "story_points": -1}, {**story, "title": ""}, {**story, "upstream": {"ambiguity_score": 1.5}},
                {**story, "blocker_count": -2}):
        assert status([bad]) == 422
    assert status([{**story, "story_id": f"S{i}"} for i in range(200)]) == 200  # the largest backlog allowed
    assert status([{**story, "story_id": f"S{i}"} for i in range(201)]) == 422
