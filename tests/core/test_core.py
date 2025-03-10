"""Tests for core functionality."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


class MockEmbeddingModel:
    """Mock implementation of EmbeddingModel for testing."""

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name
        self.device = device
        # No SentenceTransformer initialization here

    def embed_query(self, text):
        return [0.1] * 384

    def get_embedding_dimension(self):
        return 384

    def __call__(self, input):
        return [[0.1] * 384] * len(input)

    def embed_with_retries(self, input, **kwargs):
        return self.__call__(input)


# Mock RAGPipeline class
class MockRAGPipeline:
    """Mock implementation of RAGPipeline for testing."""

    def __init__(self, vectorstore=None, llm=None, top_k=None):
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k

        # Add attributes for the new modular implementation
        self._retriever = MagicMock()
        self._formatter = MagicMock()
        self._generator = MagicMock()

        # Configure mock behavior
        self._retriever.retrieve.return_value = [{"content": "test content", "metadata": {"source": "test.txt"}}]
        self._formatter.format_context.return_value = "Formatted context: test content"
        self._generator.generate.return_value = "Test response"

    def query(self, query, conversation_id=None):
        """Mock query method that returns predefined response for test query."""
        docs = [{"content": "test content", "metadata": {"source": "test.txt"}}]
        return {
            "response": "Test response",
            "documents": docs,
            "confidence": 0.9,
            "conversation_id": conversation_id or "test-id",
            "query": query,
        }


# Now patch the EmbeddingModel before importing main
with patch("llm_rag.models.embeddings.EmbeddingModel", MockEmbeddingModel):
    try:
        # Import main directly from the module
        from llm_rag.main import main
        from llm_rag.utils.test_utils import (
            create_test_data_directory,
            is_ci_environment,
        )
    except ImportError:
        # For linter, define dummy functions
        def main():
            """Dummy main function for linter."""
            pass

        def is_ci_environment():
            """Dummy function for linter."""
            return True

        def create_test_data_directory():
            """Dummy function for linter."""
            return Path("./test_data")


def test_main_output(capsys):
    """Test if main function prints the expected output.

    This test works in two modes:
    1. In CI environments: Uses mocks for directory and file operations
    2. In local environments: Creates a real test directory with sample files
    """
    # Create test data directory if not in CI
    if not is_ci_environment():
        test_dir = create_test_data_directory()
        print(f"Created test directory at {test_dir}")

    # Mock command line arguments
    argv = ["main.py", "--data-dir", "./test_data", "--query", "test query"]
    with patch.object(sys, "argv", argv):
        # Mock ChromaDB client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client.get_or_create_collection.return_value = mock_collection

        # Mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [{"content": "test content", "metadata": {"source": "test.txt"}}]
        mock_vector_store.similarity_search.return_value = [
            {"content": "test content", "metadata": {"source": "test.txt"}}
        ]

        # Mock DirectoryLoader
        mock_loader = MagicMock()
        mock_loader.load.return_value = [{"content": "test content", "metadata": {}}]

        # Mock SentenceTransformer
        mock_transformer = MagicMock()
        mock_transformer.encode.return_value = [[0.1] * 384]  # Embedding size

        # Create LlamaCpp mock
        mock_llama = MagicMock()
        mock_llama.invoke.return_value = "Test response"
        mock_llama.return_value = {"choices": [{"text": "Test response"}]}

        # Create a mock for the LLM with all required methods
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "Test response"
        mock_llm.invoke.return_value = "Test response"
        mock_llm._call.return_value = "Test response"
        mock_llm.callbacks = []

        # Determine which patches to apply based on environment
        patches = [
            patch("chromadb.PersistentClient", return_value=mock_client),
            patch(
                "llm_rag.vectorstore.chroma.ChromaVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "langchain_community.llms.LlamaCpp.__init__",
                return_value=None,
            ),
            patch(
                "langchain_community.llms.LlamaCpp.__call__",
                return_value="Test response",
            ),
            patch(
                "langchain_community.llms.LlamaCpp.invoke",
                return_value="Test response",
            ),
            patch(
                "sentence_transformers.SentenceTransformer.__init__",
                return_value=None,
            ),
            patch(
                "sentence_transformers.SentenceTransformer",
                return_value=mock_transformer,
            ),
            patch(
                "sentence_transformers.util.load_file_path",
                return_value=None,
            ),
            patch("os.makedirs"),
            patch("sys.exit"),
            patch("llama_cpp.Llama", return_value=mock_llama),
            patch(
                "llm_rag.main.CustomLlamaCpp",
                return_value=mock_llm,
            ),
            # Add patch for RAGPipeline
            patch(
                "llm_rag.main.RAGPipeline",
                MockRAGPipeline,
            ),
        ]

        # Add CI-specific patches
        if is_ci_environment():
            # In CI, we need to mock the directory loader and file existence
            patches.extend(
                [
                    patch(
                        "llm_rag.document_processing.loaders.DirectoryLoader",
                        return_value=mock_loader,
                    ),
                    patch(
                        "os.path.exists",
                        side_effect=lambda path: (path != "./models/llama-2-7b-chat.gguf" and path != "./test_data"),
                    ),
                    patch(
                        "pathlib.Path.exists",
                        return_value=True,
                    ),
                    patch(
                        "pathlib.Path.is_dir",
                        return_value=True,
                    ),
                ]
            )
        else:
            # In local environment, we only need to mock the model path
            patches.extend(
                [
                    patch(
                        "os.path.exists",
                        side_effect=lambda path: (path != "./models/llama-2-7b-chat.gguf"),
                    ),
                ]
            )

        # Apply all patches using a nested with statement
        with nested_patch(*patches):
            main()

        # Get captured output
        captured = capsys.readouterr()
        # Check for expected output patterns
        assert "QUERY: test query" in captured.out
        assert "ANSWER:" in captured.out


def nested_patch(*patches):
    """Create a nested context manager for multiple patches.

    Args:
        *patches: Patch objects to apply

    Returns:
        A context manager that applies all patches
    """

    class NestedPatch:
        def __init__(self, patches):
            self.patches = patches

        def __enter__(self):
            self.patchers = []
            for p in self.patches:
                patcher = p.__enter__()
                self.patchers.append(patcher)
            return self.patchers

        def __exit__(self, *args, **kwargs):
            for p in reversed(self.patches):
                p.__exit__(*args, **kwargs)

    return NestedPatch(patches)
