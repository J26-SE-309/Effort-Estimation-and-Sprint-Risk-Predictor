def _story(story_id, **upstream):
    return {"story_id": story_id, "title": "Add retry logic", "upstream": upstream}


def test_estimate_returns_one_prediction_per_story_in_order(client):
    payload = {"project_id": "SYN", "stories": [_story("S-1"), _story("S-2")]}
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [p["story_id"] for p in predictions] == ["S-1", "S-2"]
    lower, upper = predictions[0]["prediction_interval"]["lower"], predictions[0]["prediction_interval"]["upper"]
    assert lower <= predictions[0]["predicted_story_points"] <= upper


def test_missing_upstream_groups_are_reported_as_not_used(client):
    payload = {"project_id": "SYN", "stories": [_story("S-1"), _story("S-2", ambiguity_score=0.7)]}
    groups = [p["feature_groups_used"] for p in client.post("/api/v1/estimate", json=payload).json()["predictions"]]
    assert "requirement_quality" not in groups[0]
    assert "requirement_quality" in groups[1]
    assert "traceability" not in groups[1]


def test_models_lists_the_arena_configurations(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert len(response.json()) == 9


def test_endpoints_not_built_yet_say_so(client):
    for path in ("/api/v1/risk", "/api/v1/recommend", "/api/v1/compare"):
        assert client.post(path).status_code == 501
