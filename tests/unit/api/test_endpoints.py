"""Unit tests for API endpoints."""

from fastapi.testclient import TestClient
from llm_rag.api.main import app

client = TestClient(app)


def test_query_endpoint():
    """Test the query endpoint."""
    response = client.post("/query", json={"query": "test query"})
    assert response.status_code == 200
    assert "result" in response.json()
