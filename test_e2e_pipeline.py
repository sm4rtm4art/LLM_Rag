"""End-to-end test for the RAG pipeline.

This script tests the end-to-end functionality of the RAG pipeline to ensure
that the core functionality works correctly.
"""

from unittest.mock import MagicMock

# Import directly from the modules where the classes are defined
from llm_rag.rag.pipeline.base import RAGPipeline
from llm_rag.rag.pipeline.context import create_formatter
from llm_rag.rag.pipeline.conversational import ConversationalRAGPipeline


def test_rag_pipeline():
    """Test the RAG pipeline end-to-end."""
    # Create mock objects
    mock_vectorstore = MagicMock()
    mock_llm = MagicMock()

    # Configure mock behavior
    mock_documents = [
        {"content": "Test document 1", "metadata": {"source": "test1.txt"}},
        {"content": "Test document 2", "metadata": {"source": "test2.txt"}},
    ]
    mock_vectorstore.similarity_search.return_value = mock_documents
    mock_llm.invoke.return_value.content = "This is a test response."

    # Create pipeline with initialize=False to prevent component creation
    pipeline = RAGPipeline(
        vectorstore=mock_vectorstore,
        llm=mock_llm,
        top_k=2,
    )

    # Test individual component methods
    try:
        # Create test documents
        test_docs = [
            {"content": "Test document 1"},
            {"content": "Test document 2"},
        ]

        # Test format_context directly
        mock_formatter = create_formatter(_test=True)
        context = mock_formatter.format_context(test_docs)
        assert isinstance(context, str)
        print("✅ Formatter.format_context() works")

        # Test the components from the pipeline
        docs = pipeline._retriever.retrieve("test query")
        assert len(docs) == 2
        print("✅ Pipeline._retriever.retrieve() works")

        context = pipeline._formatter.format_context(test_docs)
        assert isinstance(context, str)
        print("✅ Pipeline._formatter.format_context() works")

        response = pipeline._generator.generate(query="test query", context=context)
        assert response == "This is a test response."
        print("✅ Pipeline._generator.generate() works")
    except Exception as e:
        print(f"❌ Error testing component methods: {e}")
        raise

    print("✅ RAGPipeline components work")


def test_conversational_pipeline():
    """Test the conversational RAG pipeline end-to-end."""
    # Create mock objects
    mock_vectorstore = MagicMock()
    mock_llm = MagicMock()

    # Configure mock behavior
    mock_documents = [
        {"content": "Test document 1", "metadata": {"source": "test1.txt"}},
        {"content": "Test document 2", "metadata": {"source": "test2.txt"}},
    ]
    mock_vectorstore.similarity_search.return_value = mock_documents

    # Important: Set up both invoke and predict for backward compatibility
    mock_llm.invoke.return_value.content = "This is a test response."
    mock_llm.predict = MagicMock(return_value="This is a test response.")

    # Create pipeline
    pipeline = ConversationalRAGPipeline(
        vectorstore=mock_vectorstore,
        llm=mock_llm,
        top_k=2,
    )
    print("✅ ConversationalRAGPipeline created")

    # Test individual component methods
    try:
        # Create test documents
        test_docs = [
            {"content": "Test document 1"},
            {"content": "Test document 2"},
        ]

        # Test the components from the pipeline
        docs = pipeline._retriever.retrieve("test query")
        assert len(docs) == 2
        print("✅ Pipeline._retriever.retrieve() works")

        context = pipeline._formatter.format_context(test_docs)
        assert isinstance(context, str)
        print("✅ Pipeline._formatter.format_context() works")

        response = pipeline._generator.generate(query="test query", context=context)
        assert response == "This is a test response."
        print("✅ Pipeline._generator.generate() works")
    except Exception as e:
        print(f"❌ Error testing component methods: {e}")
        raise

    print("✅ ConversationalRAGPipeline components work")


if __name__ == "__main__":
    print("\n=== Testing RAGPipeline ===")
    test_rag_pipeline()

    print("\n=== Testing ConversationalRAGPipeline ===")
    test_conversational_pipeline()

    print("\n✅ All tests passed!")
