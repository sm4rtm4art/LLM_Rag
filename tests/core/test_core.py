"""Tests for core functionality."""

import sys
from unittest.mock import MagicMock, patch


# Create a mock EmbeddingModel class
class MockEmbeddingModel:
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


# Now patch the EmbeddingModel before importing main
with patch("llm_rag.models.embeddings.EmbeddingModel", MockEmbeddingModel):
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
        ), patch("os.makedirs"), patch("sys.exit"), patch("os.path.exists", return_value=True):
            main()

        # Get captured output
        captured = capsys.readouterr()
        assert "Test response" in captured.out
