from fastapi.testclient import TestClient

from graphrag.api import app


def test_dashboard_and_query_roundtrip():
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert b"GraphRAG Console" in home.content
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["ok"] is True
        assert body["nodes"] >= 1
        result = client.post(
            "/query",
            json={"question": "Who is the CEO of the company that acquired DeepMind?"},
        )
        assert result.status_code == 200
        payload = result.json()
        assert "Sundar Pichai" in payload["answer"]
        graph = client.get("/graph")
        assert graph.status_code == 200
        assert graph.json()["nodes"]
