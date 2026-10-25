"""The prediction API with the real trained models (ml-engine/models/arena-v1)."""

import time

import pytest

TEAM = {"velocity_mean": 20, "velocity_variance": 16, "closed_sprints": 8, "spillover_rate": 0.3}


def _story(story_id, **fields):
    return {"story_id": story_id, "title": "Add retry logic to the payment webhook", **fields}


def _backlog(n):
    return [_story(f"S-{i}", description=f"Retry failed webhook calls up to {i % 5 + 1} times.",
                   story_points=[1, 2, 3, 5, 8][i % 5]) for i in range(n)]


def test_estimate_returns_one_real_prediction_per_story_in_order(client):
    payload = {"project_id": "SYN", "stories": [_story("S-1"), _story("S-2", story_points=5)]}
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200
    body = response.json()
    predictions = body["predictions"]
    assert [p["story_id"] for p in predictions] == ["S-1", "S-2"]
    for p in predictions:
        assert p["prediction_interval"]["lower"] <= p["predicted_story_points"] <= p["prediction_interval"]["upper"]
        assert 0 <= p["spillover_probability"] <= 1 and 0 <= p["confidence_score"] <= 1
        assert p["model_version"].startswith("arena-v1/") and p["model_version"] != "stub"
        assert p["prediction_id"]
    # a new project with unknown team history gets the pooled winner (FR11)
    assert body["selection_mode"] == "auto" and body["configuration_id"] == "fasttext-lightgbm"


def test_each_story_gets_its_own_estimate_as_if_predicted_alone(client):
    small = _story("S-1", title="Fix typo on the login page")
    big = _story("S-2", title="Rewrite the scheduler to support multi-tenant resource quotas with preemption",
                 description="Large redesign of the allocator, master and agents; needs a design document, a data "
                             "migration and extensive tests.")
    both = client.post("/api/v1/estimate", json={"project_id": "MESOS", "stories": [small, big]}).json()
    alone = [client.post("/api/v1/estimate", json={"project_id": "MESOS", "stories": [s]}).json()["predictions"][0]
             for s in (small, big)]
    points = [p["predicted_story_points"] for p in both["predictions"]]
    assert points[0] != points[1]
    assert points == [p["predicted_story_points"] for p in alone]


def test_missing_upstream_groups_are_flagged_as_degraded(client):
    payload = {"project_id": "SYN", "stories": [
        _story("S-1"),
        _story("S-2", upstream={"ambiguity_score": 0.7}),
        _story("S-3", upstream={"traceability_coverage_pct": 0.33, "unlinked_artifact_count": 2,
                                "has_linked_tests": False}),
    ]}
    predictions = client.post("/api/v1/estimate", json=payload).json()["predictions"]
    assert "requirement_quality" in predictions[0]["degraded_feature_groups"]
    assert predictions[0]["feature_sources"]["requirement_quality"] == "proxy"
    assert "requirement_quality" in predictions[1]["feature_groups_used"]
    assert predictions[1]["feature_sources"]["requirement_quality"] == "component"
    assert predictions[0]["feature_sources"]["traceability"] == "missing"
    assert predictions[2]["feature_sources"]["traceability"] == "component"
    assert "traceability" not in predictions[1]["feature_groups_used"]


def test_synthetic_upstream_signals_are_reported_as_synthetic(client):
    signals = {"ambiguity_score": 0.2, "ac_completeness_score": 0.8, "has_linked_tests": True,
               "traceability_coverage_pct": 0.67, "unlinked_artifact_count": 1, "source": "synthetic"}
    prediction = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": [
        _story("S-1", upstream=signals)]}).json()["predictions"][0]
    for group in ("requirement_quality", "acceptance_criteria", "traceability"):
        assert prediction["feature_sources"][group] == "synthetic"
        assert group in prediction["feature_groups_used"]


def test_the_confidence_score_says_what_lowered_it(client):
    cold = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": [_story("S-1")]}).json()[
        "predictions"][0]
    parts = cold["confidence_breakdown"]
    assert [p["reason"] for p in parts["penalties"]] == ["cold_start", "unseen_project", "missing_group"]
    assert parts["penalties"][2]["detail"].startswith("2 feature group(s)")  # traceability, sprint context
    assert parts["data_support"] == pytest.approx(0.7 * 0.7 * 0.85**2, abs=1e-3)
    assert cold["confidence_score"] == pytest.approx(
        parts["data_support"] * (parts["effort_certainty"] + parts["risk_certainty"]) / 2, abs=1e-3)
    signals = {"traceability_coverage_pct": 0.67, "unlinked_artifact_count": 1, "has_linked_tests": True}
    warm = client.post("/api/v1/estimate", json={"project_id": "SYN", "team_context": TEAM, "stories": [
        _story("S-1", upstream=signals)], "sprint_context": {"length_days": 14}}).json()["predictions"][0][
        "confidence_breakdown"]
    # an unseen project with enough closed sprints: the penalty eases from x 0.7 to x 0.85
    assert [p["reason"] for p in warm["penalties"]] == ["unseen_project"] and warm["data_support"] == 0.85
    assert warm["penalties"][0]["factor"] == 0.85 and parts["penalties"][1]["factor"] == 0.7


def test_each_action_says_what_it_is_expected_to_do_to_the_risk(client):
    story = _story("S-1", description="Make the search fast and user-friendly, etc.", blocker_count=2,
                   dep_out_degree=2, story_points=8)
    team = {**TEAM, "velocity_mean": 10}
    prediction = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": [
        story, _story("S-2", story_points=8)], "team_context": team, "sprint_context": {"length_days": 14},
        "focus_story_ids": ["S-1"]}).json()["predictions"][0]
    assert prediction["sprint_risk_level"] != "low" and prediction["recommendations"]
    effects = [rec["expected_effect"] for rec in prediction["recommendations"]]
    assert all(e is None or (0 <= e["spillover_probability"] <= 1 and e["sprint_risk_level"] in
                             ("low", "medium", "high")) for e in effects)
    resolve = next(r for r in prediction["recommendations"] if r["action"] == "resolve_dependency")
    assert resolve["expected_effect"] is not None
    if sum(e is not None for e in effects) > 1:
        assert prediction["risk_if_all_done"]["spillover_probability"] <= prediction["spillover_probability"]


def test_focus_stories_are_predicted_with_the_rest_of_the_sprint_as_context(client):
    backlog = _backlog(4)
    whole = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": backlog,
                                                   "team_context": TEAM}).json()["predictions"]
    focus = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": backlog, "team_context": TEAM,
                                                   "focus_story_ids": ["S-2"]}).json()["predictions"]
    assert [p["story_id"] for p in focus] == ["S-2"]
    assert focus[0]["spillover_probability"] == whole[2]["spillover_probability"]
    # only the focus story is recorded, with the rest of the sprint (1 + 2 + 5 points) as committed work
    recorded = client.get("/api/v1/projects/SYN/predictions", params={"story_id": "S-2"}).json()["predictions"]
    assert len(recorded) == 2
    seen = client.get(f"/api/v1/predictions/{focus[0]['prediction_id']}").json()["features"]
    assert seen["sprint_committed_points"] == 8
    unknown = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": backlog,
                                                     "focus_story_ids": ["S-9"]})
    assert unknown.status_code == 422


def test_reasons_and_recommendations_follow_the_explanation(client):
    story = _story("S-1", description="Make the search fast and user-friendly, etc.", blocker_count=2,
                   dep_out_degree=2)
    body = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": [story],
                                                  "team_context": TEAM}).json()
    prediction = body["predictions"][0]
    factors = [r["factor"] for r in prediction["key_risk_reasons"]]
    assert 0 < len(factors) <= 3 and all(r["direction"] == "increases" for r in prediction["key_risk_reasons"])
    assert prediction["explanation_method"] == "TreeSHAP"
    for rec in prediction["recommendations"]:  # every action points at a reason (or at the size, for splitting)
        assert rec["triggered_by"] in factors or rec["action"] == "split_story"


def test_pinned_configuration_and_unknown_pin(client):
    payload = {"project_id": "SYN", "stories": [_story("S-1")], "pinned_configuration": "tfidf-svm"}
    body = client.post("/api/v1/estimate", json=payload).json()
    assert body["selection_mode"] == "pinned" and body["configuration_id"] == "tfidf-svm"
    assert body["predictions"][0]["explanation_method"] == "feature-group occlusion"
    payload["pinned_configuration"] = "no-such-model"
    assert client.post("/api/v1/estimate", json=payload).status_code == 422


def test_a_stored_pin_overrides_the_router_until_removed(client):
    assert client.put("/api/v1/projects/SYN/pin", json={"configuration_id": "tfidf-rf"}).status_code == 200
    assert client.get("/api/v1/projects/SYN/pin").json()["configuration_id"] == "tfidf-rf"
    body = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": [_story("S-1")]}).json()
    assert body["configuration_id"] == "tfidf-rf" and body["selection_mode"] == "pinned"
    assert client.delete("/api/v1/projects/SYN/pin").json()["configuration_id"] is None
    body = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": [_story("S-1")]}).json()
    assert body["selection_mode"] == "auto"
    assert client.put("/api/v1/projects/SYN/pin", json={"configuration_id": "nope"}).status_code == 422


def test_sprint_risk_needs_capacity_and_flags_overcommitment(client):
    backlog = {"project_id": "SYN", "stories": _backlog(12)}
    unknown = client.post("/api/v1/risk", json=backlog).json()["sprint"]
    assert unknown["overcommit_probability"] is None and unknown["expected_stories_at_risk"] >= 0
    small_team = client.post("/api/v1/risk", json={**backlog, "team_context": {**TEAM, "velocity_mean": 8}}).json()
    big_team = client.post("/api/v1/risk", json={**backlog, "team_context": {**TEAM, "velocity_mean": 200}}).json()
    assert small_team["sprint"]["overcommit_probability"] > 0.9 > 0.1 > big_team["sprint"]["overcommit_probability"]
    assert small_team["sprint"]["sprint_risk_level"] == "high"
    assert [r["action"] for r in small_team["sprint"]["recommendations"]] == ["reduce_sprint_scope"]


def test_recommend_returns_story_and_sprint_recommendations(client):
    body = client.post("/api/v1/recommend", json={"project_id": "SYN", "stories": _backlog(3),
                                                   "team_context": TEAM}).json()
    assert [s["story_id"] for s in body["stories"]] == ["S-0", "S-1", "S-2"]
    assert isinstance(body["sprint"], list)


def test_compare_shows_configurations_side_by_side(client):
    payload = {"project_id": "SYN", "stories": [_story("S-1")], "configurations": ["fasttext-lightgbm", "tfidf-svm"]}
    body = client.post("/api/v1/compare", json=payload).json()
    assert [r["configuration_id"] for r in body["results"]] == ["fasttext-lightgbm", "tfidf-svm"]
    assert client.post("/api/v1/compare", json={**payload, "configurations": ["nope"]}).status_code == 422


def test_models_lists_the_leaderboard_and_the_planned_configuration(client):
    body = client.get("/api/v1/models").json()
    assert body["pooled_winner"] == "fasttext-lightgbm"
    by_id = {m["configuration_id"]: m for m in body["configurations"]}
    assert len(by_id) == 9
    assert by_id["fasttext-lightgbm"]["status"] == "available" and by_id["fasttext-lightgbm"]["loaded"]
    assert by_id["distilbert"]["status"] in ("planned", "available", "not installed")
    assert "mae" in by_id["tfidf-svm"]["metrics"]


def test_feedback_and_outcomes_are_recorded_for_known_predictions(client):
    prediction = client.post("/api/v1/estimate", json={"project_id": "SYN", "stories": [_story("S-1")]}).json()[
        "predictions"][0]
    feedback = client.post("/api/v1/feedback", json={"prediction_id": prediction["prediction_id"],
                                                     "decision": "adjust", "adjusted_story_points": 5})
    assert feedback.status_code == 201 and feedback.json()["recorded"]
    outcome = client.post("/api/v1/outcomes", json={"prediction_id": prediction["prediction_id"],
                                                    "completed_in_sprint": False, "actual_story_points": 8})
    assert outcome.status_code == 201 and outcome.json()["recorded"]
    unknown = client.post("/api/v1/feedback", json={"prediction_id": "missing", "decision": "accept"})
    assert unknown.status_code == 404


def test_predictions_still_work_when_the_database_is_down(client_without_database):
    body = client_without_database.post("/api/v1/estimate", json={"project_id": "SYN", "stories": [_story("S-1")]})
    assert body.status_code == 200 and body.json()["predictions"][0]["predicted_story_points"] > 0
    feedback = client_without_database.post("/api/v1/feedback", json={"prediction_id": "x", "decision": "accept"})
    assert feedback.status_code == 201 and feedback.json()["recorded"] is False


@pytest.mark.parametrize("stories", [50])
def test_a_50_story_backlog_is_answered_within_two_seconds(client, stories):
    """NFR1 on this machine for one user (the 10-concurrent-user load test runs against the container)."""
    payload = {"project_id": "SYN", "stories": _backlog(stories), "team_context": TEAM}
    client.post("/api/v1/estimate", json=payload)  # warm-up
    times = []
    for _ in range(5):
        started = time.perf_counter()
        assert client.post("/api/v1/estimate", json=payload).status_code == 200
        times.append(time.perf_counter() - started)
    assert sorted(times)[-1] < 2.0
