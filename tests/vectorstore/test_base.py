"""Tests for the vector store base class."""

import pytest
from llm_rag.vectorstore.base import VectorStore


class MockVectorStore(VectorStore):
    """Mock implementation of VectorStore for testing."""

    async def add_documents(self, documents):
        return None

    async def query(self, query, top_k=5):
        return [{"id": 1, "text": "test document"}]


@pytest.mark.asyncio
async def test_vector_store_interface():
    """Test the vector store interface."""
    store = MockVectorStore()
    result = await store.query("test query")
    assert isinstance(result, list)
    assert len(result) > 0
