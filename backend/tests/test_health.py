def test_health_reports_the_service_and_its_database(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "effort-estimation",
        "version": "0.1.0",
        "database": "unavailable",
    }
