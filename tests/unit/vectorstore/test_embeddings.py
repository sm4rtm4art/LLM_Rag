"""Tests for embedding function wrapper."""

import numpy as np

from llm_rag.vectorstore.chroma import EmbeddingFunctionWrapper


def test_embedding_function_wrapper() -> None:
    """Test the embedding function wrapper."""
    wrapper = EmbeddingFunctionWrapper()

    # Test single document
    result = wrapper(["This is a test document"])
    assert isinstance(result, list), "Outer result should be a list"
    assert len(result) == 1, "Should have one document embedding"
    assert isinstance(result[0], np.ndarray), "Inner result should be numpy array"
    assert result[0].dtype.kind in "if", "Array should contain numeric values"

    # Test multiple documents
    results = wrapper(["Doc 1", "Doc 2"])
    assert len(results) == 2, "Should have two document embeddings"
    for embedding in results:
        assert isinstance(embedding, np.ndarray), "Each embedding should be numpy array"
        assert embedding.dtype.kind in "if", "Arrays should contain numeric values"


def test_embedding_function_custom_model() -> None:
    """Test with a different model."""
    wrapper = EmbeddingFunctionWrapper(model_name="paraphrase-MiniLM-L6-v2")
    result = wrapper(["Test"])

    assert isinstance(result, list), "Result must be a list"
    assert len(result) == 1, "Expected single result"
    assert isinstance(result[0], np.ndarray), "Embedding must be numpy array"
    assert result[0].dtype.kind in "if", "Array must contain numeric values"
