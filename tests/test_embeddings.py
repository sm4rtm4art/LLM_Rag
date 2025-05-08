"""Tests for the embedding model."""

import pytest

from llm_rag.models.embeddings import EmbeddingModel


@pytest.fixture
def embedding_model():
    """Create an embedding model for testing."""
    return EmbeddingModel(model_name='all-MiniLM-L6-v2')


def test_embed_query(embedding_model):
    """Test embedding a single query."""
    # Given
    query = 'This is a test query'

    # When
    embedding = embedding_model.embed_query(query)

    # Then
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)
    assert len(embedding) == embedding_model.get_embedding_dimension()


def test_embed_documents(embedding_model):
    """Test embedding multiple documents."""
    # Given
    documents = [
        'This is the first document',
        'This is the second document',
        'This is the third document',
    ]

    # When
    embeddings = embedding_model.embed_documents(documents)

    # Then
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(documents)
    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
        assert len(embedding) == embedding_model.get_embedding_dimension()


def test_get_embedding_dimension(embedding_model):
    """Test getting the embedding dimension."""
    # When
    dim = embedding_model.get_embedding_dimension()

    # Then
    assert isinstance(dim, int)
    assert dim > 0
    # all-MiniLM-L6-v2 has 384-dimensional embeddings
    assert dim == 384
