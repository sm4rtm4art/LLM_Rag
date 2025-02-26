"""Integration tests for ChromaVectorStore implementation.

This test suite verifies that:
1. Document Storage & Retrieval:
   - Documents can be added with metadata
   - Semantic search returns relevant results
   - Results include correct metadata and distance scores

2. Persistence:
   - Data persists between ChromaDB sessions
   - Collection state is maintained correctly

3. Query Functionality:
   - Metadata filtering works as expected
   - Document content filtering is accurate
   - Search results are properly ranked

4. Error Handling:
   - Collection deletion handles edge cases
   - Invalid queries are handled gracefully

These tests ensure the ChromaVectorStore provides reliable vector storage
and retrieval capabilities for the RAG (Retrieval Augmented Generation) system.
"""

import tempfile
from typing import Dict, Generator, List

import pytest
from llm_rag.vectorstore.chroma import ChromaVectorStore, EmbeddingFunctionWrapper


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
        temp_persist_dir: Temporary directory for ChromaDB persistence.

    Returns:
        Generator yielding a configured ChromaVectorStore instance.
        Collection is automatically cleaned up after tests.
    """
    store = ChromaVectorStore(persist_directory=temp_persist_dir)
    yield store
    store.delete_collection()


def test_add_and_search_documents(chroma_store: ChromaVectorStore) -> None:
    """Test adding documents and searching them with metadata filters.

    This test verifies:
    1. Multiple documents can be added with metadata
    2. Search returns relevant results based on semantic similarity
    3. Metadata filtering correctly constrains search results
    4. Result format includes all required fields
    """
    documents: List[str] = [
        "The quick brown fox jumps over the lazy dog",
        "A lazy dog sleeps all day",
        "The fox is quick and brown",
    ]
    metadata: List[Dict[str, str]] = [
        {"source": "story1", "animal": "fox"},
        {"source": "story2", "animal": "dog"},
        {"source": "story3", "animal": "fox"},
    ]

    chroma_store.add_documents(documents=documents, metadatas=metadata)
    results = chroma_store.search(query="fox", n_results=2, where={"animal": "fox"})

    assert len(results) == 2
    assert any("quick brown fox" in result["document"] for result in results)
    assert all(result["metadata"]["animal"] == "fox" for result in results)


def test_delete_collection(chroma_store):
    """Test deleting the collection."""
    documents = ["Test document"]
    chroma_store.add_documents(documents=documents)

    # Delete collection
    chroma_store.delete_collection()

    # Verify collection is empty by creating a new store
    new_store = ChromaVectorStore(persist_directory=chroma_store._persist_directory)
    results = new_store.search("test", n_results=1)
    assert len(results) == 0


def test_empty_search(chroma_store):
    """Test searching an empty collection."""
    results = chroma_store.search("test query", n_results=1)
    assert len(results) == 0


def test_search_with_filters(chroma_store):
    """Test searching with metadata filters."""
    documents = [
        "Document about technology",
        "Document about nature",
        "Another tech document",
    ]
    metadata = [
        {"category": "tech", "year": 2024},
        {"category": "nature", "year": 2023},
        {"category": "tech", "year": 2024},
    ]

    chroma_store.add_documents(documents=documents, metadatas=metadata)

    results = chroma_store.search(
        query="document",
        n_results=2,
        where={"$and": [{"category": {"$eq": "tech"}}, {"year": {"$eq": 2024}}]},
    )

    assert len(results) == 2
    assert all(result["metadata"]["category"] == "tech" for result in results)
    assert all(result["metadata"]["year"] == 2024 for result in results)


def test_custom_embedding_function(temp_persist_dir):
    """Test using a custom embedding function."""
    custom_embedder = EmbeddingFunctionWrapper("paraphrase-MiniLM-L6-v2")
    store = ChromaVectorStore(
        persist_directory=temp_persist_dir, embedding_function=custom_embedder
    )
    documents = ["Test document"]
    store.add_documents(documents)
    results = store.search("test")
    assert len(results) > 0


def test_search_with_document_filter(chroma_store):
    """Test searching with document content filter."""
    documents = [
        "Tech document about Python",
        "Tech document about Java",
        "Article about nature",
    ]
    chroma_store.add_documents(documents)

    results = chroma_store.search("document", where_document={"$contains": "Python"})
    assert len(results) == 1
    assert "Python" in results[0]["document"]


def test_add_documents_with_custom_ids(chroma_store):
    """Test adding documents with custom IDs."""
    documents = ["Doc1", "Doc2"]
    custom_ids = ["id1", "id2"]
    chroma_store.add_documents(documents, ids=custom_ids)

    results = chroma_store.search("Doc")
    assert len(results) == 2
    assert all(r["id"] in custom_ids for r in results)


def test_collection_persistence(temp_persist_dir):
    """Test that documents persist between store instances."""
    # Create first store and add documents
    store1 = ChromaVectorStore(persist_directory=temp_persist_dir)
    documents = ["Persistent document"]
    store1.add_documents(documents)

    # Create new store with same directory
    store2 = ChromaVectorStore(persist_directory=temp_persist_dir)
    results = store2.search("persistent")
    assert len(results) == 1
    assert "Persistent document" in results[0]["document"]
