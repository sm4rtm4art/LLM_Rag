"""Unit tests for API endpoints."""

from fastapi.testclient import TestClient

from llm_rag.api.main import app

client = TestClient(app)


def test_query_endpoint():
    """Test the query endpoint."""
    response = client.post('/query', json={'query': 'test query'})
    assert response.status_code == 200

    # Check for the expected fields in the response
    response_data = response.json()
    assert 'query' in response_data
    assert 'response' in response_data
    assert 'retrieved_documents' in response_data

    # Check that the query matches what we sent
    assert response_data['query'] == 'test query'
