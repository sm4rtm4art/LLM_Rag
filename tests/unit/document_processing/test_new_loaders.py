#!/usr/bin/env python3
"""Unit tests for the new modular document loaders.

This module contains tests for the refactored document loading system,
including the registry, factory functions, and new loader implementations.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from llm_rag.document_processing.loaders import CSVLoader, LoaderRegistry, TextFileLoader
from llm_rag.document_processing.loaders.directory_loader import DirectoryLoader, load_document


class TestLoaderRegistry:
    """Test cases for the LoaderRegistry class."""

    def test_registry_initialization(self):
        """Test that the registry is initialized correctly."""
        test_registry = LoaderRegistry()
        assert test_registry._loaders == {}
        assert test_registry._extension_mapping == {}

    def test_register_loader(self):
        """Test registering a loader class."""
        test_registry = LoaderRegistry()

        # Register a loader
        test_registry.register(TextFileLoader, extensions=["txt", "md"])

        # Check that it was registered correctly
        assert "TextFileLoader" in test_registry._loaders
        assert test_registry._loaders["TextFileLoader"] == TextFileLoader
        assert test_registry._extension_mapping[".txt"] == "TextFileLoader"
        assert test_registry._extension_mapping[".md"] == "TextFileLoader"

    def test_get_loader_class(self):
        """Test getting a loader class by name."""
        test_registry = LoaderRegistry()
        test_registry.register(TextFileLoader)

        # Get the loader class
        loader_class = test_registry.get_loader_class("TextFileLoader")
        assert loader_class == TextFileLoader

        # Test with non-existent loader
        with pytest.raises(KeyError):
            test_registry.get_loader_class("NonExistentLoader")

    def test_create_loader(self):
        """Test creating a loader instance."""
        test_registry = LoaderRegistry()
        test_registry.register(TextFileLoader)

        # Create a loader instance
        loader = test_registry.create_loader("TextFileLoader", file_path="test.txt")
        assert isinstance(loader, TextFileLoader)
        assert loader.file_path == Path("test.txt")

    def test_get_loader_for_extension(self):
        """Test getting a loader name for a file extension."""
        test_registry = LoaderRegistry()
        test_registry.register(TextFileLoader, extensions=["txt"])

        # Get loader for extension
        loader_name = test_registry.get_loader_for_extension("txt")
        assert loader_name == "TextFileLoader"

        # Test with leading dot
        loader_name = test_registry.get_loader_for_extension(".txt")
        assert loader_name == "TextFileLoader"

        # Test with non-existent extension
        loader_name = test_registry.get_loader_for_extension("xyz")
        assert loader_name is None

    def test_create_loader_for_file(self):
        """Test creating a loader for a file based on its extension."""
        test_registry = LoaderRegistry()
        test_registry.register(TextFileLoader, extensions=["txt"])

        # Create loader for file
        loader = test_registry.create_loader_for_file("test.txt")
        assert isinstance(loader, TextFileLoader)

        # Test with non-existent extension
        loader = test_registry.create_loader_for_file("test.xyz")
        assert loader is None


class TestFactoryFunctions:
    """Test cases for document loader factory functions."""

    def test_load_document(self):
        """Test loading a document using the load_document function."""
        # Mock file existence check
        with patch("pathlib.Path.exists", return_value=True):
            # Mock registry
            with patch("llm_rag.document_processing.loaders.directory_loader.registry") as mock_registry:
                # Setup mock loader
                mock_loader = MagicMock()
                mock_loader.load_from_file.return_value = [{"content": "Test content", "metadata": {}}]
                mock_registry.create_loader_for_file.return_value = mock_loader

                # Call load_document
                documents = load_document("test.txt")

                # Check that registry was called correctly
                mock_registry.create_loader_for_file.assert_called_once()

                # Check that loader was called correctly
                mock_loader.load_from_file.assert_called_once()

                # Check result
                assert len(documents) == 1
                assert documents[0]["content"] == "Test content"

    def test_load_document_file_not_found(self):
        """Test load_document when the file doesn't exist."""
        # Mock file existence check
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                load_document("nonexistent.txt")

    def test_load_document_no_loader(self):
        """Test load_document when no loader is found for the file type."""
        # Mock file existence check
        with patch("pathlib.Path.exists", return_value=True):
            # Mock registry
            with patch("llm_rag.document_processing.loaders.directory_loader.registry") as mock_registry:
                mock_registry.create_loader_for_file.return_value = None

                # Call load_document
                with pytest.raises(ValueError):
                    load_document("test.unknown")

    def test_load_documents_from_directory(self):
        """Test loading documents from a directory."""
        # Mock directory existence check
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                # Mock glob
                with patch("pathlib.Path.glob") as mock_glob:
                    mock_glob.return_value = [Path("file1.txt"), Path("file2.txt")]

                    # Mock load_document
                    with patch(
                        "llm_rag.document_processing.loaders.directory_loader.load_document"
                    ) as mock_load_document:
                        mock_load_document.side_effect = [
                            [{"content": "Content 1", "metadata": {}}],
                            [{"content": "Content 2", "metadata": {}}],
                        ]

                        # Call load_documents_from_directory
                        loader = DirectoryLoader("test_dir")
                        documents = loader.load()

                        # Check that load_document was called correctly
                        assert mock_load_document.call_count == 2

                        # Check results
                        assert len(documents) == 2
                        assert documents[0]["content"] == "Content 1"
                        assert documents[1]["content"] == "Content 2"

    def test_load_documents_from_directory_not_found(self):
        """Test load_documents_from_directory when the directory doesn't exist."""
        # Mock directory existence check
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(NotADirectoryError):
                DirectoryLoader("nonexistent").load()


class TestTextFileLoader:
    """Test cases for the new TextFileLoader class."""

    def test_initialization(self):
        """Test initialization of the TextFileLoader."""
        loader = TextFileLoader(file_path="test.txt")
        assert loader.file_path == Path("test.txt")

    def test_load(self):
        """Test loading a text file."""
        # Create a loader with a file path
        loader = TextFileLoader(file_path="test.txt")

        # Mock the load_from_file method
        with patch.object(loader, "load_from_file") as mock_load_from_file:
            mock_load_from_file.return_value = [{"content": "Test content", "metadata": {}}]

            # Call load
            documents = loader.load()

            # Check that load_from_file was called correctly
            mock_load_from_file.assert_called_once_with(Path("test.txt"))

            # Check returned documents
            assert documents == [{"content": "Test content", "metadata": {}}]

    def test_load_no_file_path(self):
        """Test load when no file path was provided."""
        loader = TextFileLoader()
        with pytest.raises(ValueError):
            loader.load()

    def test_load_from_file(self):
        """Test loading from a file."""
        loader = TextFileLoader()

        # Mock opening the file
        file_content = "This is a test file.\nIt has multiple lines.\n"
        m = mock_open(read_data=file_content)

        with patch("builtins.open", m):
            # Call load_from_file
            documents = loader.load_from_file("test.txt")

            # Check that the file was opened correctly
            m.assert_called_once_with(Path("test.txt"), "r", encoding="utf-8")

            # Check returned documents
            assert len(documents) == 1
            assert documents[0]["content"] == file_content
            assert documents[0]["metadata"]["source"] == "test.txt"
            assert documents[0]["metadata"]["filename"] == "test.txt"
            assert documents[0]["metadata"]["filetype"] == "text"


class TestCSVLoader:
    """Test cases for the new CSVLoader class."""

    def test_initialization(self):
        """Test initialization of the CSVLoader."""
        loader = CSVLoader(
            file_path="test.csv",
            content_columns=["col1", "col2"],
            metadata_columns=["col3"],
            delimiter=";",
            use_pandas=False,
        )
        assert loader.file_path == Path("test.csv")
        assert loader.content_columns == ["col1", "col2"]
        assert loader.metadata_columns == ["col3"]
        assert loader.delimiter == ";"
        assert not loader.use_pandas

    def test_load(self):
        """Test loading a CSV file."""
        # Create a loader with a file path
        loader = CSVLoader(file_path="test.csv")

        # Mock the load_from_file method
        with patch.object(loader, "load_from_file") as mock_load_from_file:
            mock_load_from_file.return_value = [
                {"content": "row1", "metadata": {}},
                {"content": "row2", "metadata": {}},
            ]

            # Call load
            documents = loader.load()

            # Check that load_from_file was called correctly
            mock_load_from_file.assert_called_once_with(Path("test.csv"))

            # Check returned documents
            assert len(documents) == 2
            assert documents[0]["content"] == "row1"
            assert documents[1]["content"] == "row2"

    def test_load_from_file_csv_module(self):
        """Test loading from a CSV file using the csv module."""
        loader = CSVLoader(use_pandas=False)

        # Mock the DictReader
        with patch("csv.DictReader") as mock_dict_reader:
            mock_dict_reader.return_value = [{"col1": "val1", "col2": "val2"}, {"col1": "val3", "col2": "val4"}]

            # Mock opening the file
            m = mock_open()
            with patch("builtins.open", m):
                # Call load_from_file
                documents = loader.load_from_file("test.csv")

                # Check that the file was opened correctly
                m.assert_called_once_with(Path("test.csv"), newline="", encoding="utf-8")

                # Check returned documents
                assert len(documents) == 2
                # Content should be "col1: val1, col2: val2" and "col1: val3, col2: val4"
                # But we can't test the exact strings due to dict ordering
                assert "col1: val1" in documents[0]["content"] or "col1: val1" in documents[0]["content"]
                assert "col2: val2" in documents[0]["content"] or "col2: val2" in documents[0]["content"]
                assert documents[0]["metadata"]["filetype"] == "csv"
                assert documents[0]["metadata"]["row_index"] == 0
                assert documents[1]["metadata"]["row_index"] == 1
