"""Tests for the model factory module."""

import unittest
from unittest.mock import MagicMock, patch

from src.llm_rag.models.factory import ModelBackend, ModelFactory


class TestModelBackend(unittest.TestCase):
    """Tests for the ModelBackend enum."""

    def test_model_backend_values(self):
        """Test that the ModelBackend enum has the expected values."""
        self.assertEqual(ModelBackend.LLAMA_CPP.value, "llama_cpp")
        self.assertEqual(ModelBackend.HUGGINGFACE.value, "huggingface")


class TestModelFactory(unittest.TestCase):
    """Tests for the ModelFactory class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects for the LLM classes
        self.mock_llama_cpp = MagicMock()
        self.mock_huggingface_llm = MagicMock()

        # Create patches for the imports
        self.llama_cpp_patcher = patch("src.llm_rag.main.CustomLlamaCpp", return_value=self.mock_llama_cpp)
        self.huggingface_patcher = patch(
            "src.llm_rag.models.huggingface.HuggingFaceLLM", return_value=self.mock_huggingface_llm
        )

        # Start the patchers
        self.mock_llama_cpp_class = self.llama_cpp_patcher.start()
        self.mock_huggingface_class = self.huggingface_patcher.start()

        # Patch os.path.exists to always return True
        self.path_exists_patcher = patch("os.path.exists", return_value=True)
        self.mock_path_exists = self.path_exists_patcher.start()

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patchers
        self.llama_cpp_patcher.stop()
        self.huggingface_patcher.stop()
        self.path_exists_patcher.stop()

    def test_create_model_llama_cpp(self):
        """Test creating a llama-cpp model."""
        # Call the factory method
        model = ModelFactory.create_model(
            model_path_or_name="models/llama.gguf",
            backend=ModelBackend.LLAMA_CPP,
            max_tokens=256,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            n_ctx=2048,
        )

        # Check that the model was created with the correct parameters
        self.assertEqual(model, self.mock_llama_cpp)
        self.mock_llama_cpp_class.assert_called_once_with(
            model_path="models/llama.gguf", max_tokens=256, temperature=0.8, top_p=0.9, repeat_penalty=1.2, n_ctx=2048
        )

    def test_create_model_huggingface(self):
        """Test creating a Hugging Face model."""
        # Call the factory method
        model = ModelFactory.create_model(
            model_path_or_name="meta-llama/Llama-2-7b-hf",
            backend=ModelBackend.HUGGINGFACE,
            device="cuda",
            max_tokens=256,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            trust_remote_code=True,
        )

        # Check that the model was created with the correct parameters
        self.assertEqual(model, self.mock_huggingface_llm)
        self.mock_huggingface_class.assert_called_once_with(
            model_name="meta-llama/Llama-2-7b-hf",
            device="cuda",
            max_tokens=256,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            trust_remote_code=True,
        )

    def test_create_model_with_string_backend(self):
        """Test creating a model with a string backend."""
        # Call the factory method with a string backend
        model = ModelFactory.create_model(model_path_or_name="models/llama.gguf", backend="llama_cpp", max_tokens=256)

        # Check that the model was created with the correct parameters
        self.assertEqual(model, self.mock_llama_cpp)
        self.mock_llama_cpp_class.assert_called_once()

    def test_create_model_invalid_backend(self):
        """Test creating a model with an invalid backend."""
        # Call the factory method with an invalid backend
        with self.assertRaises(ValueError):
            ModelFactory.create_model(model_path_or_name="models/llama.gguf", backend="invalid_backend")

    def test_create_llama_cpp_model_file_not_found(self):
        """Test creating a llama-cpp model with a non-existent file."""
        # Patch os.path.exists to return False
        self.mock_path_exists.return_value = False

        # Call the factory method
        with self.assertRaises(FileNotFoundError):
            ModelFactory._create_llama_cpp_model(model_path="non_existent_file.gguf")


if __name__ == "__main__":
    unittest.main()
