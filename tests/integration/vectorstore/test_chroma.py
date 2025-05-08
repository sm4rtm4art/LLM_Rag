"""Integration tests for ChromaDB functionality."""

import os
import tempfile
from typing import Generator, List

import numpy as np
import pytest

from llm_rag.vectorstore.chroma import (
    ChromaVectorStore,
    EmbeddingFunctionWrapper,
)


class MockEmbeddingFunction:
    """Mock embedding function for CI testing.

    This avoids downloading the actual model in CI environments.
    """

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Return fixed embeddings with the correct dimensions (384).

        Args:
            input: List of texts to embed

        Returns:
            List of mock embeddings with fixed values but correct dimensions
        """
        # Return a fixed embedding for each text (384 dimensions)
        return [list(np.zeros(384) + 0.1) for _ in input]


@pytest.fixture
def temp_persist_dir() -> Generator[str, None, None]:
    """Create a temporary directory for ChromaDB persistence.

    Returns:
        Generator yielding the temporary directory path.
        Directory is automatically cleaned up after tests.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def chroma_store(temp_persist_dir: str) -> Generator[ChromaVectorStore, None, None]:
    """Create a ChromaVectorStore instance for testing.

    Args:
        temp_persist_dir: Temporary directory for ChromaDB persistence

    Returns:
        Generator yielding a ChromaVectorStore instance
    """
    # Check if we're in a CI environment
    in_ci = os.environ.get('GITHUB_ACTIONS') == 'true'

    # Use mock embedding function in CI to avoid downloading models
    embedding_function = MockEmbeddingFunction() if in_ci else None

    store = ChromaVectorStore(persist_directory=temp_persist_dir, embedding_function=embedding_function)
    yield store
    store.delete_collection()


def test_add_and_search_documents(chroma_store: ChromaVectorStore) -> None:
    """Test adding documents and searching."""
    # Test documents
    documents = [
        'The quick brown fox jumps over the lazy dog',
        'A different document about cats',
        'Another document about the quick brown fox',
    ]

    # Test metadata
    metadata = [
        {'animal': 'fox', 'action': 'jump'},
        {'animal': 'cat', 'action': 'sleep'},
        {'animal': 'fox', 'action': 'run'},
    ]

    # Add documents
    chroma_store.add_documents(documents=documents, metadatas=metadata)

    # Search with no filters
    results = chroma_store.search(query='fox', n_results=2)
    assert len(results) == 2
    assert any('quick brown fox' in result['document'] for result in results)

    # Search with metadata filter
    chroma_store.add_documents(documents=documents, metadatas=metadata)
    results = chroma_store.search(query='fox', n_results=2, where={'animal': 'fox'})

    assert len(results) == 2
    assert any('quick brown fox' in result['document'] for result in results)


def test_delete_collection(chroma_store):
    """Test deleting a collection."""
    # Add some documents
    documents = ['Test document 1', 'Test document 2']
    chroma_store.add_documents(documents=documents)

    # Delete the collection
    chroma_store.delete_collection()

    # Verify collection is empty by creating a new store
    new_store = ChromaVectorStore(persist_directory=chroma_store._persist_directory)
    results = new_store.search('test', n_results=1)
    assert len(results) == 0


def test_empty_search(chroma_store):
    """Test searching an empty collection."""
    results = chroma_store.search('test', n_results=1)
    assert len(results) == 0


def test_search_with_filters(chroma_store):
    """Test searching with complex filters."""
    # Add documents with metadata
    documents = [
        'Technical document about Python programming',
        'Technical document about JavaScript programming',
        'News article about politics',
        'Blog post about cooking',
    ]

    metadata = [
        {'category': 'tech', 'language': 'Python', 'year': 2024},
        {'category': 'tech', 'language': 'JavaScript', 'year': 2024},
        {'category': 'news', 'topic': 'politics', 'year': 2023},
        {'category': 'lifestyle', 'topic': 'cooking', 'year': 2022},
    ]

    chroma_store.add_documents(documents=documents, metadatas=metadata)

    # Test with complex filter
    results = chroma_store.search(
        query='document',
        n_results=2,
        where={'$and': [{'category': {'$eq': 'tech'}}, {'year': {'$eq': 2024}}]},
    )

    assert len(results) == 2
    assert all('Technical document' in r['document'] for r in results)


def test_custom_embedding_function(temp_persist_dir):
    """Test using a custom embedding function."""
    # Skip this test in CI environment since it requires downloading models
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        pytest.skip('Skipping custom embedding function test in CI environment')

    custom_embedder = EmbeddingFunctionWrapper('paraphrase-MiniLM-L6-v2')
    store = ChromaVectorStore(persist_directory=temp_persist_dir, embedding_function=custom_embedder)
    documents = ['Test document']
    store.add_documents(documents)
    results = store.search('test')

    assert len(results) == 1
    assert results[0]['document'] == 'Test document'


def test_search_with_document_filter(chroma_store):
    """Test searching with document content filter."""
    documents = [
        'Python programming guide',
        'JavaScript tutorial',
        'Data science with Python',
    ]

    chroma_store.add_documents(documents)

    results = chroma_store.search('document', where_document={'$contains': 'Python'})
    # The filter matches two documents:
    # "Python programming guide" and "Data science with Python"
    assert len(results) == 2

    # Verify both Python documents are in the results
    result_docs = [r['document'] for r in results]
    assert 'Python programming guide' in result_docs
    assert 'Data science with Python' in result_docs


def test_add_documents_with_custom_ids(chroma_store):
    """Test adding documents with custom IDs."""
    documents = ['Document one', 'Document two']
    ids = ['custom-id-1', 'custom-id-2']

    chroma_store.add_documents(documents=documents, ids=ids)
    results = chroma_store.search('document', n_results=2)

    assert len(results) == 2
    assert results[0]['id'] in ids
    assert results[1]['id'] in ids


def test_collection_persistence(temp_persist_dir):
    """Test that collections persist between instances."""
    # Create a store and add documents
    store1 = ChromaVectorStore(persist_directory=temp_persist_dir)
    documents = ['Persistent document test']
    store1.add_documents(documents=documents)

    # Create a new store with the same persist directory
    store2 = ChromaVectorStore(persist_directory=temp_persist_dir)
    results = store2.search('persistent', n_results=1)

    assert len(results) == 1
    assert 'Persistent document test' in results[0]['document']
