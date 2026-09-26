def test_health_reports_the_service_its_database_and_its_models(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert {k: body[k] for k in ("status", "service", "version", "database", "models")} == {
        "status": "ok",
        "service": "effort-estimation",
        "version": "0.1.0",
        "database": "ok",
        "models": "ok",
    }
    assert "fasttext-lightgbm" in body["loaded_configurations"]  # the pooled winner is loaded at start-up


def test_health_when_the_database_is_down(client_without_database):
    assert client_without_database.get("/health").json()["database"] == "unavailable"
