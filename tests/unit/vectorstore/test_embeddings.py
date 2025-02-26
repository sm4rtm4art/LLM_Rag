"""Tests for embedding function wrapper."""


import numpy as np

from llm_rag.vectorstore.chroma import EmbeddingFunctionWrapper


def test_embedding_function_wrapper():
    """Test the embedding function wrapper."""
    wrapper = EmbeddingFunctionWrapper()

    # Test single document
    result = wrapper(["This is a test document"])
    assert isinstance(result, list), "Outer result should be a list"
    assert len(result) == 1, "Should have one document embedding"
    assert isinstance(result[0], (list, np.ndarray)), "Inner result should be list or numpy array"

    if isinstance(result[0], list):
        assert all(isinstance(x, (int, float)) for x in result[0])
    else:
        assert result[0].dtype.kind in "if"

    # Test multiple documents
    results = wrapper(["Doc 1", "Doc 2"])
    assert len(results) == 2, "Should have two document embeddings"
    for embedding in results:
        assert isinstance(embedding, (list, np.ndarray))
        if isinstance(embedding, list):
            assert all(isinstance(x, (int, float)) for x in embedding)
        else:
            assert embedding.dtype.kind in "if"


def test_embedding_function_custom_model() -> None:
    """Test with a different model."""
    wrapper = EmbeddingFunctionWrapper(model_name="paraphrase-MiniLM-L6-v2")
    result = wrapper(["Test"])

    # Check basic structure
    assert isinstance(result, list), "Result must be a list"
    assert len(result) == 1, "Expected single result"

    # Check embedding format
    embedding = result[0]
    assert isinstance(embedding, (list, np.ndarray)), "Embedding must be list or ndarray"

    # Check values based on type
    if isinstance(embedding, np.ndarray):
        assert embedding.dtype.kind in "if", "NumPy array must contain numeric values"
    else:
        assert isinstance(embedding, list), "Embedding must be a list at this point"
        assert all(isinstance(x, (int, float)) for x in embedding), "All values must be numeric"
