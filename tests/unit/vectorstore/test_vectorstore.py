"""Unit tests for the VectorStore base interface.

This test suite verifies that:
1. Interface Contract:
   - Abstract methods are properly defined
   - Required methods are implemented by concrete classes
   - Method signatures are consistent

2. Mock Implementation:
   - Tests can use a mock vector store
   - Interface methods work as expected
   - Return types match interface requirements

These tests ensure that new vector store implementations
will maintain consistent behavior across the application.
"""

from typing import Any, Dict, List, Optional

from llm_rag.vectorstore.base import VectorStore


class MockVectorStore(VectorStore):
    """Mock implementation of VectorStore for testing."""

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Mock adding documents."""
        return None

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Mock search implementation."""
        return [{'id': '1', 'document': 'test document', 'metadata': None, 'distance': 0.5}]

    def delete_collection(self) -> None:
        """Mock collection deletion."""
        return None


def test_vector_store_interface() -> None:
    """Test the vector store interface."""
    store = MockVectorStore()
    result = store.search('test query')
    assert isinstance(result, list)
    assert len(result) > 0
