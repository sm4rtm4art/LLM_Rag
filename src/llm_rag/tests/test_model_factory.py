"""Tests for the ModelFactory class."""

import sys
import unittest
from unittest.mock import MagicMock, patch

from src.llm_rag.models.factory import ModelBackend, ModelFactory


class TestModelFactory(unittest.TestCase):
    """Tests for the ModelFactory class."""

    def test_model_backend_enum(self):
        """Test the ModelBackend enum."""
        self.assertEqual(ModelBackend.LLAMA_CPP, "llama_cpp")
        self.assertEqual(ModelBackend.HUGGINGFACE, "huggingface")
        self.assertEqual(ModelBackend.OLLAMA, "ollama")

    @patch("src.llm_rag.models.factory.ModelFactory._create_llama_cpp_model")
    def test_create_llama_cpp_model(self, mock_create_llama_cpp):
        """Test creating a llama_cpp model."""
        # Set up the mock
        mock_model = MagicMock()
        mock_create_llama_cpp.return_value = mock_model

        # Call the method
        model = ModelFactory.create_model(
            model_path_or_name="model.gguf",
            backend=ModelBackend.LLAMA_CPP,
        )

        # Assert the mock was called with the right parameters
        mock_create_llama_cpp.assert_called_once_with(
            model_path="model.gguf",
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        self.assertEqual(model, mock_model)

    @patch("src.llm_rag.models.factory.ModelFactory._create_huggingface_model")
    def test_create_huggingface_model(self, mock_create_huggingface):
        """Test creating a huggingface model."""
        # Set up the mock
        mock_model = MagicMock()
        mock_create_huggingface.return_value = mock_model

        # Call the method
        model = ModelFactory.create_model(
            model_path_or_name="meta-llama/Llama-2-7b-chat-hf",
            backend=ModelBackend.HUGGINGFACE,
            device="cuda",
        )

        # Assert the mock was called with the right parameters
        mock_create_huggingface.assert_called_once_with(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            device="cuda",
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        self.assertEqual(model, mock_model)

    @patch("src.llm_rag.models.factory.ModelFactory._create_ollama_model")
    def test_create_ollama_model(self, mock_create_ollama):
        """Test creating an Ollama model."""
        # Set up the mock
        mock_model = MagicMock()
        mock_create_ollama.return_value = mock_model

        # Call the method
        model = ModelFactory.create_model(
            model_path_or_name="llama3",
            backend=ModelBackend.OLLAMA,
            temperature=0.8,
        )

        # Assert the mock was called with the right parameters
        mock_create_ollama.assert_called_once_with(
            model_name="llama3",
            max_tokens=512,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        self.assertEqual(model, mock_model)

    def test_unsupported_backend(self):
        """Test creating a model with an unsupported backend."""
        with self.assertRaises(ValueError):
            ModelFactory.create_model(
                model_path_or_name="model",
                backend="unsupported_backend",
            )


class TestCreateOllamaModelWithMocks(unittest.TestCase):
    """Tests for the _create_ollama_model with import mocks."""

    @patch("src.llm_rag.models.factory.logger")
    def test_create_ollama_with_langchain_ollama(self, mock_logger):
        """Test creating an Ollama model with langchain_ollama."""
        # Mock the imports
        with patch.dict(
            "sys.modules",
            {
                "langchain_ollama": MagicMock(),
            },
        ):
            # Create a mock for OllamaLLM
            mock_ollama_llm = MagicMock()
            # Add the mock to the langchain_ollama module
            sys.modules["langchain_ollama"].OllamaLLM = mock_ollama_llm

            # Call the method
            # This should try to import langchain_ollama.OllamaLLM first
            ModelFactory._create_ollama_model(model_name="llama3")

            # Assert the mock was called
            mock_ollama_llm.assert_called_once()
            # Check that the correct log message was generated
            mock_logger.info.assert_any_call("Using langchain_ollama.OllamaLLM")

    @patch("src.llm_rag.models.factory.logger")
    def test_create_ollama_with_langchain_community(self, mock_logger):
        """Test creating an Ollama model with langchain_community."""
        # Mock the imports
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "langchain_ollama":
                raise ImportError("Module not found")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with patch.dict(
                "sys.modules",
                {
                    "langchain_community": MagicMock(),
                    "langchain_community.llms": MagicMock(),
                    "langchain_community.llms.ollama": MagicMock(),
                },
            ):
                # Create a mock for OllamaLLM
                mock_ollama_llm = MagicMock()
                # Add the mock to the langchain_community.llms.ollama module
                sys.modules["langchain_community.llms.ollama"].OllamaLLM = mock_ollama_llm

                # Call the method
                # This should try to import langchain_community.llms.ollama.OllamaLLM
                ModelFactory._create_ollama_model(model_name="llama3")

                # Assert the mock was called
                mock_ollama_llm.assert_called_once()
                # Check that the correct log message was generated
                mock_logger.info.assert_any_call("Using langchain_community.llms.ollama.OllamaLLM")


if __name__ == "__main__":
    unittest.main()
