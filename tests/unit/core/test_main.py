"""Unit tests for core functionality."""

import sys
from unittest.mock import MagicMock, patch

from llm_rag.main import main


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

        # Apply all mocks
        with patch("chromadb.PersistentClient", return_value=mock_client), patch(
            "llm_rag.vectorstore.chroma.ChromaVectorStore", return_value=mock_vector_store
        ), patch("llm_rag.document_processing.loaders.DirectoryLoader", return_value=mock_loader), patch(
            "langchain_community.llms.LlamaCpp.__init__", return_value=None
        ), patch("langchain_community.llms.LlamaCpp.__call__", return_value="Test response"), patch(
            "langchain_community.llms.LlamaCpp.invoke", return_value="Test response"
        ), patch("sentence_transformers.SentenceTransformer.__init__", return_value=None), patch(
            "sentence_transformers.SentenceTransformer", return_value=mock_transformer
        ), patch("sentence_transformers.util.load_file_path", return_value=None), patch("os.makedirs"), patch(
            "sys.exit"
        ), patch("os.path.exists", return_value=True):
            main()

        # Get captured output
        captured = capsys.readouterr()
        assert "Test response" in captured.out
