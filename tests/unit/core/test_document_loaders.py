"""Unit tests for document loaders module."""

import tempfile
import unittest
from inspect import isabstract
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from llm_rag.document_processing.loaders import (
    CSVLoader,
    DirectoryLoader,
    DocumentLoader,
    TextFileLoader,
)


class TestDocumentLoader(unittest.TestCase):
    """Test cases for the abstract DocumentLoader class."""

    def test_abstract_class(self):
        """Test that DocumentLoader is an abstract class."""
        # Check that the class is abstract
        self.assertTrue(isabstract(DocumentLoader))

        # Check that the load method is abstract
        self.assertTrue(hasattr(DocumentLoader, "load"))
        load_method = DocumentLoader.load
        self.assertTrue(getattr(load_method, "__isabstractmethod__", False))


class TestTextFileLoader(unittest.TestCase):
    """Test cases for the TextFileLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file_path = Path(self.temp_dir.name) / "test.txt"

        # Write test content to the file
        with open(self.temp_file_path, "w", encoding="utf-8") as f:
            f.write("This is a test document.\nIt has multiple lines.\n")

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_init_with_nonexistent_file(self):
        """Test initialization with a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            TextFileLoader("nonexistent_file.txt")

    def test_init_with_string_path(self):
        """Test initialization with a string path."""
        loader = TextFileLoader(str(self.temp_file_path))
        self.assertEqual(loader.file_path, self.temp_file_path)

    def test_init_with_path_object(self):
        """Test initialization with a Path object."""
        loader = TextFileLoader(self.temp_file_path)
        self.assertEqual(loader.file_path, self.temp_file_path)

    def test_load(self):
        """Test loading a text file."""
        loader = TextFileLoader(self.temp_file_path)
        documents = loader.load()

        # Check that we got one document
        self.assertEqual(len(documents), 1)

        # Check document content
        content = "This is a test document.\nIt has multiple lines.\n"
        self.assertEqual(documents[0]["content"], content)

        # Check document metadata
        metadata = documents[0]["metadata"]
        self.assertEqual(metadata["source"], str(self.temp_file_path))
        self.assertEqual(metadata["filename"], "test.txt")
        self.assertEqual(metadata["filetype"], "text")


class TestCSVLoader(unittest.TestCase):
    """Test cases for the CSVLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file_path = Path(self.temp_dir.name) / "test.csv"

        # Create a test DataFrame and save it as CSV
        self.test_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["Document 1", "Document 2", "Document 3"],
                "content": ["Content 1", "Content 2", "Content 3"],
                "author": ["Author 1", "Author 2", "Author 3"],
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            }
        )
        self.test_df.to_csv(self.temp_file_path, index=False)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_init_with_nonexistent_file(self):
        """Test initialization with a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            CSVLoader("nonexistent_file.csv")

    def test_load_all_columns_as_content(self):
        """Test loading a CSV file with all columns as content."""
        loader = CSVLoader(self.temp_file_path)
        documents = loader.load()

        # Check that we got one document per row
        self.assertEqual(len(documents), 3)

        # Check first document content
        content = documents[0]["content"]
        self.assertIn("id: 1", content)
        self.assertIn("title: Document 1", content)
        self.assertIn("content: Content 1", content)
        self.assertIn("author: Author 1", content)
        self.assertIn("date: 2023-01-01", content)

        # Check document metadata
        metadata = documents[0]["metadata"]
        self.assertEqual(metadata["source"], str(self.temp_file_path))
        self.assertEqual(metadata["filename"], "test.csv")
        self.assertEqual(metadata["filetype"], "csv")

    def test_load_with_specific_content_columns(self):
        """Test loading a CSV file with specific content columns."""
        loader = CSVLoader(self.temp_file_path, content_columns=["title", "content"])
        documents = loader.load()

        # Check first document content
        content = documents[0]["content"]
        self.assertIn("title: Document 1", content)
        self.assertIn("content: Content 1", content)
        self.assertNotIn("id: 1", content)
        self.assertNotIn("author: Author 1", content)

    def test_load_with_metadata_columns(self):
        """Test loading a CSV file with metadata columns."""
        loader = CSVLoader(
            self.temp_file_path, content_columns=["title", "content"], metadata_columns=["id", "author", "date"]
        )
        documents = loader.load()

        # Check first document content
        content = documents[0]["content"]
        self.assertIn("title: Document 1", content)
        self.assertIn("content: Content 1", content)
        self.assertNotIn("id: 1", content)

        # Check document metadata
        metadata = documents[0]["metadata"]
        self.assertEqual(metadata["id"], "1")
        self.assertEqual(metadata["author"], "Author 1")
        self.assertEqual(metadata["date"], "2023-01-01")

    def test_load_with_missing_values(self):
        """Test loading a CSV file with missing values."""
        # Create a DataFrame with missing values
        df_with_na = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "title": ["Document 1", None, "Document 3"],
                "content": ["Content 1", "Content 2", None],
            }
        )

        # Save to a temporary file
        na_file_path = Path(self.temp_dir.name) / "test_na.csv"
        df_with_na.to_csv(na_file_path, index=False)

        # Load the file
        loader = CSVLoader(na_file_path)
        documents = loader.load()

        # Check that missing values are not included
        self.assertNotIn("title:", documents[1]["content"])
        self.assertNotIn("content:", documents[2]["content"])


class TestDirectoryLoader(unittest.TestCase):
    """Test cases for the DirectoryLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create a subdirectory
        self.subdir_path = self.dir_path / "subdir"
        self.subdir_path.mkdir()

        # Create test text files
        self.text_file1 = self.dir_path / "file1.txt"
        with open(self.text_file1, "w", encoding="utf-8") as f:
            f.write("Content of file 1")

        self.text_file2 = self.dir_path / "file2.txt"
        with open(self.text_file2, "w", encoding="utf-8") as f:
            f.write("Content of file 2")

        # Create a text file in the subdirectory
        self.subdir_file = self.subdir_path / "subfile.txt"
        with open(self.subdir_file, "w", encoding="utf-8") as f:
            f.write("Content of subdir file")

        # Create a CSV file
        self.csv_file = self.dir_path / "data.csv"
        csv_data = {"id": [1, 2], "content": ["CSV content 1", "CSV content 2"]}
        pd.DataFrame(csv_data).to_csv(self.csv_file, index=False)

        # Create an unsupported file type
        self.unsupported_file = self.dir_path / "unsupported.xyz"
        with open(self.unsupported_file, "w", encoding="utf-8") as f:
            f.write("This file type is not supported")

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_init_with_nonexistent_directory(self):
        """Test initialization with a non-existent directory."""
        with self.assertRaises(NotADirectoryError):
            DirectoryLoader("nonexistent_directory")

    def test_init_with_file_path(self):
        """Test initialization with a file path instead of a directory."""
        with self.assertRaises(NotADirectoryError):
            DirectoryLoader(self.text_file1)

    def test_load_non_recursive(self):
        """Test loading files from a directory non-recursively."""
        loader = DirectoryLoader(self.dir_path, recursive=False)
        documents = loader.load()

        # Should load 3 documents (2 text files + 1 CSV file with 2 rows)
        self.assertEqual(len(documents), 4)

        # Check that files from subdirectory are not included
        sources = [doc["metadata"]["source"] for doc in documents]
        self.assertIn(str(self.text_file1), sources)
        self.assertIn(str(self.text_file2), sources)
        self.assertIn(str(self.csv_file), sources)
        self.assertNotIn(str(self.subdir_file), sources)

    def test_load_recursive(self):
        """Test loading files from a directory recursively."""
        loader = DirectoryLoader(self.dir_path, recursive=True)
        documents = loader.load()

        # Should load 4 documents (3 text files + 1 CSV file with 2 rows)
        self.assertEqual(len(documents), 5)

        # Check that files from subdirectory are included
        sources = [doc["metadata"]["source"] for doc in documents]
        self.assertIn(str(self.text_file1), sources)
        self.assertIn(str(self.text_file2), sources)
        self.assertIn(str(self.csv_file), sources)
        self.assertIn(str(self.subdir_file), sources)

    def test_load_with_glob_pattern(self):
        """Test loading files with a specific glob pattern."""
        loader = DirectoryLoader(self.dir_path, glob_pattern="*.txt")
        documents = loader.load()

        # Should load only the 2 text files in the main directory
        self.assertEqual(len(documents), 2)

        sources = [doc["metadata"]["source"] for doc in documents]
        self.assertIn(str(self.text_file1), sources)
        self.assertIn(str(self.text_file2), sources)
        self.assertNotIn(str(self.csv_file), sources)

    @patch("llm_rag.document_processing.loaders.TextFileLoader")
    def test_error_handling(self, mock_text_loader):
        """Test error handling when loading files."""
        # Make the TextFileLoader raise an exception
        mock_instance = MagicMock()
        mock_instance.load.side_effect = Exception("Test error")
        mock_text_loader.return_value = mock_instance

        # Redirect stdout to capture print statements
        with patch("builtins.print") as mock_print:
            loader = DirectoryLoader(self.dir_path, glob_pattern="*.txt")
            documents = loader.load()

            # Should not load any documents due to the error
            self.assertEqual(len(documents), 0)

            # Check that error was printed
            mock_print.assert_called()


if __name__ == "__main__":
    unittest.main()
