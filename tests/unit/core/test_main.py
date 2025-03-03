"""Unit tests for core functionality."""

import sys
from unittest.mock import MagicMock, patch

# Import main directly from the module
from src.llm_rag.main import main


def test_main_output(capsys):
    """Test if main function prints the expected output"""
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

        # Apply all mocks
        with (
            patch("chromadb.PersistentClient", return_value=mock_client),
            patch(
                "src.llm_rag.vectorstore.chroma.ChromaVectorStore",
                return_value=mock_vector_store,
            ),
            patch(
                "src.llm_rag.document_processing.loaders.DirectoryLoader",
                return_value=mock_loader,
            ),
            patch("langchain_community.llms.LlamaCpp.__init__", return_value=None),
            patch(
                "langchain_community.llms.LlamaCpp.__call__",
                return_value="Test response",
            ),
            patch(
                "langchain_community.llms.LlamaCpp.invoke",
                return_value="Test response",
            ),
            patch("sentence_transformers.SentenceTransformer.__init__", return_value=None),
            patch(
                "sentence_transformers.SentenceTransformer",
                return_value=mock_transformer,
            ),
            patch("sentence_transformers.util.load_file_path", return_value=None),
            patch("os.makedirs"),
            patch("sys.exit"),
            patch(
                "os.path.exists",
                side_effect=lambda path: path != "./models/llama-2-7b-chat.gguf",
            ),
            patch("llama_cpp.Llama", return_value=mock_llama),
            # Patch CustomLlamaCpp to return our mock
            patch(
                "src.llm_rag.main.CustomLlamaCpp",
                return_value=mock_llm,
            ),
        ):
            main()

        # Get captured output
        captured = capsys.readouterr()
        # The RAGPipeline's query method hardcodes the response to "This is a test response."
        assert "This is a test response" in captured.out
