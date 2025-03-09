"""Unit tests for the JSONLoader class."""

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Try to import JSONLoader
try:
    from llm_rag.document_processing.loaders import JSONLoader

    has_json_loader = True
except ImportError:
    has_json_loader = False

# Skip all tests if JSONLoader is not available
pytestmark = pytest.mark.skipif(not has_json_loader, reason="JSONLoader not available")


@pytest.mark.skipif(not has_json_loader, reason="JSONLoader not available")
class TestJSONLoader(unittest.TestCase):
    """Test cases for the JSONLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_json_path = "test.json"

    def test_initialization(self):
        """Test the initialization of JSONLoader."""
        # Arrange & Act
        loader = JSONLoader(file_path=self.test_json_path, jq_schema=".items[]", content_key="content")

        # Assert
        self.assertEqual(loader.file_path, self.test_json_path)
        self.assertEqual(loader.jq_schema, ".items[]")
        self.assertEqual(loader.content_key, "content")

    def test_initialization_default_values(self):
        """Test the initialization with default values."""
        # Arrange & Act
        loader = JSONLoader(file_path=self.test_json_path)

        # Assert
        self.assertEqual(loader.jq_schema, ".")
        self.assertIsNone(loader.content_key)

    def test_load_with_jq_filter_pattern(self):
        """Test loading JSON with jq filter pattern."""
        # Arrange
        json_data = {"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]}

        # Create mock jq
        mock_jq = MagicMock()
        mock_pattern = MagicMock()
        mock_pattern.input.return_value.all.return_value = [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]
        mock_jq.compile.return_value = mock_pattern

        with (
            patch("builtins.open", mock_open(read_data=json.dumps(json_data))),
            patch("llm_rag.document_processing.loaders.jq", mock_jq),
            patch("json.load", return_value=json_data),
        ):
            # Create loader with jq schema
            loader = JSONLoader(file_path=self.test_json_path, jq_schema=".items[]")

            # Act
            documents = loader.load()

            # Assert
            self.assertEqual(len(documents), 2)
            self.assertEqual(json.loads(documents[0]["content"]), {"id": 1, "name": "Item 1"})
            self.assertEqual(documents[0]["metadata"]["filetype"], "json")
            self.assertEqual(documents[0]["metadata"]["item_index"], 0)

            self.assertEqual(json.loads(documents[1]["content"]), {"id": 2, "name": "Item 2"})
            self.assertEqual(documents[1]["metadata"]["item_index"], 1)

            # Verify jq was used correctly
            mock_jq.compile.assert_called_once_with(".items[]")
            mock_pattern.input.assert_called_once_with(json_data)

    def test_load_with_jq_error_fallback(self):
        """Test loading JSON with jq error that falls back to normal processing."""
        # Arrange
        json_data = {"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]}

        # Create mock jq that raises an exception
        mock_jq = MagicMock()
        mock_pattern = MagicMock()
        mock_pattern.input.side_effect = Exception("jq error")
        mock_jq.compile.return_value = mock_pattern

        with (
            patch("builtins.open", mock_open(read_data=json.dumps(json_data))),
            patch("llm_rag.document_processing.loaders.jq", mock_jq),
            patch("json.load", return_value=json_data),
        ):
            # Create loader with jq schema
            loader = JSONLoader(file_path=self.test_json_path, jq_schema=".items[]")

            # Act
            documents = loader.load()

            # Assert - should contain a single document with the full json
            self.assertEqual(len(documents), 1)
            self.assertIn("items", json.loads(documents[0]["content"]))

            # Verify jq was used but failed
            mock_jq.compile.assert_called_once_with(".items[]")
            mock_pattern.input.assert_called_once_with(json_data)

    def test_load_list_json(self):
        """Test loading JSON with list structure."""
        # Arrange
        json_data = [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]

        with (
            patch("builtins.open", mock_open(read_data=json.dumps(json_data))),
            patch("json.load", return_value=json_data),
            patch("llm_rag.document_processing.loaders.jq", None),
        ):  # No jq available
            # Create loader
            loader = JSONLoader(file_path=self.test_json_path)

            # Act
            documents = loader.load()

            # Assert
            self.assertEqual(len(documents), 2)
            self.assertEqual(json.loads(documents[0]["content"]), {"id": 1, "name": "Item 1"})
            self.assertEqual(documents[0]["metadata"]["filetype"], "json")
            self.assertEqual(documents[0]["metadata"]["item_index"], 0)

            self.assertEqual(json.loads(documents[1]["content"]), {"id": 2, "name": "Item 2"})
            self.assertEqual(documents[1]["metadata"]["item_index"], 1)

    def test_load_dict_json(self):
        """Test loading JSON with dictionary structure."""
        # Arrange
        json_data = {"id": 1, "name": "Item 1", "description": "This is item 1"}

        with (
            patch("builtins.open", mock_open(read_data=json.dumps(json_data))),
            patch("json.load", return_value=json_data),
            patch("llm_rag.document_processing.loaders.jq", None),
        ):  # No jq available
            # Create loader
            loader = JSONLoader(file_path=self.test_json_path)

            # Act
            documents = loader.load()

            # Assert
            self.assertEqual(len(documents), 1)
            self.assertEqual(json.loads(documents[0]["content"]), json_data)
            self.assertEqual(documents[0]["metadata"]["filetype"], "json")
            self.assertNotIn("item_index", documents[0]["metadata"])

    def test_load_primitive_json(self):
        """Test loading JSON with primitive value."""
        # Arrange
        json_data = 42

        with (
            patch("builtins.open", mock_open(read_data=json.dumps(json_data))),
            patch("json.load", return_value=json_data),
            patch("llm_rag.document_processing.loaders.jq", None),
        ):  # No jq available
            # Create loader
            loader = JSONLoader(file_path=self.test_json_path)

            # Act
            documents = loader.load()

            # Assert
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["content"], "42")
            self.assertEqual(documents[0]["metadata"]["filetype"], "json")

    def test_extract_content_with_content_key(self):
        """Test extracting content with content_key parameter."""
        # Arrange
        data = {"id": 1, "content": "This is the content", "metadata": {"source": "test"}}

        # Create loader with content_key
        loader = JSONLoader(file_path=self.test_json_path, content_key="content")

        # Act
        content = loader._extract_content(data)

        # Assert
        self.assertEqual(content, "This is the content")

    def test_extract_content_from_dict(self):
        """Test extracting content from a dictionary."""
        # Arrange
        data = {"id": 1, "name": "Item 1", "description": "Test description"}

        # Create loader without content_key
        loader = JSONLoader(file_path=self.test_json_path)

        # Act
        content = loader._extract_content(data)

        # Assert
        self.assertEqual(json.loads(content), data)

    def test_extract_content_from_non_dict(self):
        """Test extracting content from a non-dictionary value."""
        # Arrange
        data = ["item1", "item2", "item3"]

        # Create loader
        loader = JSONLoader(file_path=self.test_json_path)

        # Act
        content = loader._extract_content(data)

        # Assert
        self.assertEqual(content, str(data))

    def test_file_not_found(self):
        """Test handling of file not found error."""
        # Arrange
        with patch("builtins.open", side_effect=FileNotFoundError()):
            # Create loader
            loader = JSONLoader(file_path=self.test_json_path)

            # Act & Assert
            with self.assertRaises(FileNotFoundError):
                loader.load()

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        # Arrange
        with (
            patch("builtins.open", mock_open(read_data="Invalid JSON")),
            patch("json.load", side_effect=json.JSONDecodeError("Invalid JSON", "", 0)),
        ):
            # Create loader
            loader = JSONLoader(file_path=self.test_json_path)

            # Act & Assert
            with self.assertRaises(json.JSONDecodeError):
                loader.load()


if __name__ == "__main__":
    unittest.main()
