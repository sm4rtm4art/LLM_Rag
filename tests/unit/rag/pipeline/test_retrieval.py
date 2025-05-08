"""Tests for retrieval components in the RAG pipeline.

This module contains comprehensive tests for the document retrieval
components of the RAG pipeline system.
"""

import unittest
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from llm_rag.rag.pipeline.retrieval import BaseRetriever, HybridRetriever, VectorStoreRetriever
from llm_rag.utils.errors import PipelineError, VectorstoreError


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self, docs=None):
        self.docs = docs or [
            {'content': 'Document 1', 'metadata': {'source': 'test1.txt'}},
            {'content': 'Document 2', 'metadata': {'source': 'test2.txt'}},
        ]
        self.similarity_search_called = False
        self.last_query = None
        self.last_k = None

    def similarity_search(self, query, k=5):
        """Mock similarity search method."""
        self.similarity_search_called = True
        self.last_query = query
        self.last_k = k
        return self.docs[:k]


class TestBaseRetriever(unittest.TestCase):
    """Tests for the BaseRetriever abstract base class."""

    def test_validate_query_valid(self):
        """Test query validation with valid input."""

        # Create a concrete subclass for testing
        class TestRetriever(BaseRetriever):
            def retrieve(self, query: str, **kwargs) -> List[Any]:
                return []

        retriever = TestRetriever()
        # This should not raise an exception
        retriever._validate_query('test query')

    def test_validate_query_invalid(self):
        """Test query validation with invalid input."""

        class TestRetriever(BaseRetriever):
            def retrieve(self, query: str, **kwargs) -> List[Any]:
                return []

        retriever = TestRetriever()

        # Test with empty query
        with self.assertRaises(PipelineError) as context:
            retriever._validate_query('')
        self.assertIn('Query must be a non-empty string', str(context.exception))

        # Test with non-string query
        with self.assertRaises(PipelineError) as context:
            retriever._validate_query(123)  # type: ignore
        self.assertIn('Query must be a non-empty string', str(context.exception))

        # Test with whitespace-only query
        with self.assertRaises(PipelineError) as context:
            retriever._validate_query('   ')
        self.assertIn('Query cannot be empty or only whitespace', str(context.exception))


class TestVectorStoreRetriever(unittest.TestCase):
    """Tests for the VectorStoreRetriever class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_vectorstore = MockVectorStore()
        self.retriever = VectorStoreRetriever(vectorstore=self.mock_vectorstore, top_k=3)

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Default initialization with custom top_k
        retriever = VectorStoreRetriever(vectorstore=self.mock_vectorstore, top_k=10)
        self.assertEqual(retriever.vectorstore, self.mock_vectorstore)
        self.assertEqual(retriever.top_k, 10)

    def test_retrieve(self):
        """Test basic document retrieval."""
        documents = self.retriever.retrieve('test query')

        # Check that the vector store was called with correct parameters
        self.assertTrue(self.mock_vectorstore.similarity_search_called)
        self.assertEqual(self.mock_vectorstore.last_query, 'test query')
        self.assertEqual(self.mock_vectorstore.last_k, 3)

        # Check returned documents
        self.assertEqual(len(documents), 2)  # Should return all docs since we only have 2
        self.assertEqual(documents[0]['content'], 'Document 1')

    def test_retrieve_with_custom_top_k(self):
        """Test retrieval with custom top_k parameter."""
        documents = self.retriever.retrieve('test query', top_k=1)

        # Check that the vector store was called with the custom top_k
        self.assertEqual(self.mock_vectorstore.last_k, 1)
        self.assertEqual(len(documents), 1)

    def test_retrieve_with_invalid_query(self):
        """Test retrieval with invalid query."""
        with self.assertRaises(PipelineError):
            self.retriever.retrieve('')

    def test_retrieve_with_vectorstore_error(self):
        """Test handling of vector store errors."""
        # Create a mock vector store that raises an exception
        error_vectorstore = MagicMock()
        error_vectorstore.similarity_search.side_effect = Exception('Vector store error')
        # Set the module attribute to simulate a langchain error
        error_vectorstore.similarity_search.side_effect.__module__ = 'langchain.vectorstores'

        retriever = VectorStoreRetriever(vectorstore=error_vectorstore)

        with self.assertRaises(VectorstoreError) as context:
            retriever.retrieve('test query')

        self.assertIn('Failed to retrieve documents from vector store', str(context.exception))

    def test_retrieve_with_other_error(self):
        """Test handling of other errors."""
        # Create a mock vector store that raises a general exception
        error_vectorstore = MagicMock()
        error_vectorstore.similarity_search.side_effect = Exception('General error')

        retriever = VectorStoreRetriever(vectorstore=error_vectorstore)

        with self.assertRaises(PipelineError) as context:
            retriever.retrieve('test query')

        self.assertIn('Document retrieval failed', str(context.exception))


class TestHybridRetriever(unittest.TestCase):
    """Tests for the HybridRetriever class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock retrievers
        self.mock_retriever1 = MagicMock()
        self.mock_retriever1.retrieve.return_value = [
            {'content': 'Doc A', 'metadata': {'source': 'source1'}},
            {'content': 'Doc B', 'metadata': {'source': 'source1'}},
        ]

        self.mock_retriever2 = MagicMock()
        self.mock_retriever2.retrieve.return_value = [
            {'content': 'Doc C', 'metadata': {'source': 'source2'}},
            {'content': 'Doc D', 'metadata': {'source': 'source2'}},
        ]

        # Create hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            retrievers=[self.mock_retriever1, self.mock_retriever2], weights=[0.7, 0.3]
        )

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Test with custom weights
        retriever = HybridRetriever(retrievers=[self.mock_retriever1, self.mock_retriever2], weights=[2, 1])
        # Weights should be normalized to sum to 1.0
        self.assertAlmostEqual(retriever.weights[0], 2 / 3)
        self.assertAlmostEqual(retriever.weights[1], 1 / 3)

        # Test with default equal weights
        retriever = HybridRetriever(retrievers=[self.mock_retriever1, self.mock_retriever2])
        self.assertAlmostEqual(retriever.weights[0], 0.5)
        self.assertAlmostEqual(retriever.weights[1], 0.5)

        # Test with invalid weights
        with self.assertRaises(ValueError):
            HybridRetriever(
                retrievers=[self.mock_retriever1, self.mock_retriever2],
                weights=[1],  # Only one weight for two retrievers
            )

    def test_retrieve(self):
        """Test basic hybrid retrieval."""
        documents = self.hybrid_retriever.retrieve('test query')

        # Both retrievers should be called
        self.mock_retriever1.retrieve.assert_called_once()
        self.mock_retriever2.retrieve.assert_called_once()

        # Should return combined results
        self.assertEqual(len(documents), 4)

    def test_retrieve_with_top_k(self):
        """Test hybrid retrieval with top_k parameter."""
        documents = self.hybrid_retriever.retrieve('test query', top_k=2)

        # Should limit results to top_k
        self.assertEqual(len(documents), 2)

    def test_retrieve_with_deduplication(self):
        """Test retrieval with deduplication."""
        # Create retrievers with overlapping results
        mock_retriever1 = MagicMock()
        mock_retriever1.retrieve.return_value = [
            {'content': 'Doc A', 'metadata': {'source': 'source1'}},
            {'content': 'Doc B', 'metadata': {'source': 'source1'}},
        ]

        mock_retriever2 = MagicMock()
        mock_retriever2.retrieve.return_value = [
            {'content': 'Doc A', 'metadata': {'source': 'source2'}},  # Duplicate content
            {'content': 'Doc C', 'metadata': {'source': 'source2'}},
        ]

        # Mock the deduplication method
        with patch('llm_rag.rag.pipeline.retrieval.HybridRetriever._deduplicate_documents') as mock_deduplicate:
            mock_deduplicate.return_value = [
                {'content': 'Doc A', 'metadata': {'source': 'source1'}},
                {'content': 'Doc B', 'metadata': {'source': 'source1'}},
                {'content': 'Doc C', 'metadata': {'source': 'source2'}},
            ]

            retriever = HybridRetriever(retrievers=[mock_retriever1, mock_retriever2])
            documents = retriever.retrieve('test query', deduplicate=True)

            # Deduplication should be called
            mock_deduplicate.assert_called_once()
            self.assertEqual(len(documents), 3)

    def test_retrieve_without_deduplication(self):
        """Test retrieval without deduplication."""
        documents = self.hybrid_retriever.retrieve('test query', deduplicate=False)

        # Should return all documents without deduplication
        self.assertEqual(len(documents), 4)

    def test_retrieve_with_retriever_failure(self):
        """Test handling of individual retriever failures."""
        # Make one retriever fail
        self.mock_retriever1.retrieve.side_effect = Exception('Retriever error')

        # Should still return results from the working retriever
        documents = self.hybrid_retriever.retrieve('test query')
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]['content'], 'Doc C')


@pytest.mark.parametrize(
    'query,expected_error',
    [
        (None, 'Query must be a non-empty string'),
        ('', 'Query must be a non-empty string'),
        (123, 'Query must be a non-empty string'),
        ('   ', 'Query cannot be empty or only whitespace'),
    ],
)
def test_retriever_validation_errors(query, expected_error):
    """Test validation errors with various invalid inputs."""
    mock_vectorstore = MockVectorStore()
    retriever = VectorStoreRetriever(vectorstore=mock_vectorstore)

    with pytest.raises(PipelineError) as excinfo:
        retriever.retrieve(query)

    assert expected_error in str(excinfo.value)


if __name__ == '__main__':
    unittest.main()
