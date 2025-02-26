"""Tests for embedding function wrapper."""

from llm_rag.vectorstore.chroma import EmbeddingFunctionWrapper


def test_embedding_function_wrapper():
    """Test the embedding function wrapper."""
    wrapper = EmbeddingFunctionWrapper()

    # Test single document
    result = wrapper(["This is a test document"])
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])

    # Test multiple documents
    results = wrapper(["Doc 1", "Doc 2"])
    assert len(results) == 2
    assert all(isinstance(x, list) for x in results)


def test_embedding_function_custom_model():
    """Test with a different model."""
    wrapper = EmbeddingFunctionWrapper(model_name="paraphrase-MiniLM-L6-v2")
    result = wrapper(["Test"])
    assert isinstance(result, list)
