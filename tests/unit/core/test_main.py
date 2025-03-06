"""Unit tests for core functionality."""

import sys
from unittest.mock import MagicMock, patch

# Mock llama_cpp module in case it's not installed
sys.modules["llama_cpp"] = MagicMock()

# Import main directly from the module
from src.llm_rag.main import main  # noqa: E402
from src.llm_rag.rag.pipeline import RAGPipeline  # noqa: E402


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
        # Make sure search returns non-empty results
        mock_vector_store.search.return_value = [{"content": "test content", "metadata": {"source": "test.txt"}}]

        # Create a mock for the LLM with all required methods
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "Test response"
        mock_llm.invoke.return_value = "Test response"
        mock_llm._call.return_value = "Test response"
        mock_llm.callbacks = []

        # Create LlamaCpp mock
        mock_llama = MagicMock()
        mock_llama.invoke.return_value = "Test response"
        mock_llama.return_value = {"choices": [{"text": "Test response"}]}

        # Mock DirectoryLoader
        mock_loader = MagicMock()
        mock_loader.load.return_value = [{"content": "test content", "metadata": {}}]

        # Mock SentenceTransformer
        mock_transformer = MagicMock()
        # Embedding size
        mock_transformer.encode.return_value = [[0.1] * 384]

        # Create a function to modify the RAGPipeline instance after it's created
        original_init = RAGPipeline.__init__

        def mock_rag_pipeline_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Override the query method to return our test response
            self.query = MagicMock(
                return_value={
                    "query": "test query",
                    "response": "Test response",
                    "documents": [{"content": "test content", "metadata": {"source": "test.txt"}}],
                }
            )

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
                # Always return True for path existence
                side_effect=lambda path: True,
            ),
            # Mock Path.exists
            patch("pathlib.Path.exists", return_value=True),
            # Mock Path.is_dir
            patch("pathlib.Path.is_dir", return_value=True),
            patch("llama_cpp.Llama", return_value=mock_llama),
            # Patch CustomLlamaCpp to return our mock
            patch(
                "src.llm_rag.main.CustomLlamaCpp",
                return_value=mock_llm,
            ),
            # Patch the RAGPipeline.__init__ method
            patch("src.llm_rag.rag.pipeline.RAGPipeline.__init__", mock_rag_pipeline_init),
        ):
            main()

        # Get captured output
        captured = capsys.readouterr()

        # Print the captured output for debugging
        print(f"Captured output: {captured.out}")

        # The RAGPipeline's query method returns a hardcoded test response
        assert "Test response" in captured.out
