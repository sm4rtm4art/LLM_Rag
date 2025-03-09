"""Unit tests for the document loaders module.

This module contains tests for the various document loaders in the loaders module,
including TextFileLoader, PDFLoader, DirectoryLoader, etc.
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from llm_rag.document_processing.loaders import (
    CSVLoader,
    DirectoryLoader,
    JSONLoader,
    PDFLoader,
    TextFileLoader,
    WebPageLoader,
)


class TestTextFileLoader:
    """Test cases for the TextFileLoader class."""

    def test_initialization(self) -> None:
        """Test the initialization of TextFileLoader."""
        # Arrange & Act
        loader = TextFileLoader(file_path="test.txt")

        # Assert
        assert loader.file_path_str == "test.txt"
        assert loader.encoding == "utf-8"  # Default encoding

    def test_load_valid_text_file(self) -> None:
        """Test loading content from a valid text file."""
        # Arrange
        file_content = "This is a test file.\nIt has multiple lines.\n"
        mock_file = mock_open(read_data=file_content)

        with patch("builtins.open", mock_file):
            # Also patch Path.exists to return True
            with patch("pathlib.Path.exists", return_value=True):
                loader = TextFileLoader(file_path="test.txt")

                # Act
                documents = loader.load()

                # Assert
                assert len(documents) == 1
                assert documents[0]["content"] == file_content
                assert documents[0]["metadata"]["source"] == "test.txt"
                assert documents[0]["metadata"]["filetype"] == "text"

    def test_load_with_custom_encoding(self) -> None:
        """Test loading with a custom encoding."""
        # Arrange
        file_content = "Encoded content"
        mock_file = mock_open(read_data=file_content)

        with patch("builtins.open", mock_file):
            # Also patch Path.exists to return True
            with patch("pathlib.Path.exists", return_value=True):
                loader = TextFileLoader(file_path="test.txt", encoding="latin-1")

                # Act
                documents = loader.load()

                # Assert
                # Check that the file was opened with the correct encoding
                open_calls = mock_file.call_args_list
                assert any(
                    call[0][0] == loader.file_path and call[1].get("encoding") == "latin-1" for call in open_calls
                )
                assert documents[0]["content"] == file_content

    def test_load_file_not_found(self) -> None:
        """Test behavior when file is not found."""
        # Arrange
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            # Also patch Path.exists to return True to bypass the check in __init__
            with patch("pathlib.Path.exists", return_value=True):
                loader = TextFileLoader(file_path="nonexistent.txt")

                # Act & Assert
                with pytest.raises(FileNotFoundError):
                    loader.load()


class TestPDFLoader:
    """Test cases for the PDFLoader class."""

    def test_initialization(self) -> None:
        """Test the initialization of PDFLoader."""
        # Arrange & Act
        loader = PDFLoader(file_path="test.pdf")

        # Assert
        assert loader.file_path_str == "test.pdf"
        assert loader.extract_images is False  # Default value
        assert loader.extract_tables is False  # Default value

    @patch("llm_rag.document_processing.loaders.fitz.open")
    def test_load_valid_pdf(self, mock_fitz_open) -> None:
        """Test loading content from a valid PDF file."""
        # Arrange
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 content"

        mock_pdf.__len__.return_value = 2
        mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]
        mock_fitz_open.return_value = mock_pdf

        loader = PDFLoader(file_path="test.pdf")

        # Act
        documents = loader.load()

        # Assert
        assert len(documents) == 2
        assert documents[0]["content"] == "Page 1 content"
        assert documents[0]["metadata"]["source"] == "test.pdf"
        assert documents[0]["metadata"]["page"] == 0
        assert documents[0]["metadata"]["filetype"] == "pdf"

        assert documents[1]["content"] == "Page 2 content"
        assert documents[1]["metadata"]["page"] == 1

    @patch("llm_rag.document_processing.loaders.fitz.open", side_effect=Exception("PDF error"))
    def test_load_invalid_pdf(self, mock_fitz_open) -> None:
        """Test behavior when PDF loading fails."""
        # Arrange
        loader = PDFLoader(file_path="bad.pdf")

        # Act & Assert
        with pytest.raises(Exception, match="Error loading PDF file"):
            loader.load()


class TestCSVLoader:
    """Test cases for the CSVLoader class."""

    def test_initialization(self) -> None:
        """Test the initialization of CSVLoader."""
        # Arrange & Act
        loader = CSVLoader(file_path="test.csv")

        # Assert
        assert loader.file_path_str == "test.csv"
        assert loader.content_columns is None  # Default value
        assert loader.metadata_columns is None  # Default value
        assert loader.delimiter == ","  # Default value

    def test_load_valid_csv(self) -> None:
        """Test loading content from a valid CSV file."""
        # Arrange
        csv_content = "name,age\nJohn,30\nJane,25\n"
        mock_file = mock_open(read_data=csv_content)

        with patch("builtins.open", mock_file):
            with patch("csv.DictReader") as mock_dict_reader:
                # Also patch Path.exists to return True
                with patch("pathlib.Path.exists", return_value=True):
                    mock_dict_reader.return_value = [{"name": "John", "age": "30"}, {"name": "Jane", "age": "25"}]

                    loader = CSVLoader(file_path="test.csv")

                    # Act
                    documents = loader.load()

                    # Assert
                    assert len(documents) == 2
                    assert documents[0]["content"] == "name: John, age: 30"
                    assert documents[0]["metadata"]["source"] == "test.csv"
                    assert documents[0]["metadata"]["filename"] == "test.csv"
                    assert documents[0]["metadata"]["filetype"] == "csv"

                    # Verify CSV reader was called with the correct delimiter
                    mock_dict_reader.assert_called_once()

    def test_load_with_custom_delimiter(self) -> None:
        """Test loading with a custom delimiter."""
        # Arrange
        csv_content = "name;age\nJohn;30\nJane;25\n"
        mock_file = mock_open(read_data=csv_content)

        with patch("builtins.open", mock_file):
            with patch("csv.DictReader") as mock_dict_reader:
                # Also patch Path.exists to return True
                with patch("pathlib.Path.exists", return_value=True):
                    mock_dict_reader.return_value = [{"name": "John", "age": "30"}, {"name": "Jane", "age": "25"}]

                    loader = CSVLoader(file_path="test.csv", delimiter=";")

                    # Act
                    loader.load()

                    # Assert
                    # Check if DictReader was called with the right delimiter
                    mock_dict_reader.assert_called_once()
                    call_args = mock_dict_reader.call_args
                    assert call_args[1].get("delimiter") == ";"


class TestJSONLoader:
    """Test cases for the JSONLoader class."""

    def test_initialization(self) -> None:
        """Test the initialization of JSONLoader."""
        # Arrange & Act
        loader = JSONLoader(file_path="test.json")

        # Assert
        assert loader.file_path_str == "test.json"
        assert loader.jq_schema == "."  # Default value
        assert loader.content_key is None  # Default value

    def test_load_valid_json(self) -> None:
        """Test loading content from a valid JSON file."""
        # Arrange
        json_content = '{"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]}'
        mock_file = mock_open(read_data=json_content)

        with patch("builtins.open", mock_file):
            with patch("json.load") as mock_json_load:
                mock_json_load.return_value = {"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]}

                loader = JSONLoader(file_path="test.json", jq_schema=".items[]")

                # Act
                with patch("llm_rag.document_processing.loaders.jq") as mock_jq:
                    mock_jq.compile.return_value.input.return_value = [
                        {"id": 1, "name": "Item 1"},
                        {"id": 2, "name": "Item 2"},
                    ]

                    documents = loader.load()

                    # Assert
                    assert len(documents) == 2
                    assert documents[0]["content"] == '{"id": 1, "name": "Item 1"}'
                    assert documents[0]["metadata"]["source"] == "test.json"
                    assert documents[0]["metadata"]["filetype"] == "json"

                    assert documents[1]["content"] == '{"id": 2, "name": "Item 2"}'


class TestWebPageLoader:
    """Test cases for the WebPageLoader class."""

    def test_initialization(self) -> None:
        """Test the initialization of WebPageLoader."""
        # Arrange & Act
        loader = WebPageLoader(url="https://example.com")

        # Assert
        assert loader.url == "https://example.com"
        assert loader.encoding == "utf-8"  # Default encoding

    @patch("llm_rag.document_processing.loaders.requests.get")
    def test_load_valid_webpage(self, mock_get) -> None:
        """Test loading content from a valid webpage."""
        # Arrange
        html_content = "<html><body><h1>Test Page</h1><p>Test content</p></body></html>"
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.headers = {"Content-Type": "text/plain"}  # Use plain text to avoid BeautifulSoup
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        loader = WebPageLoader(url="https://example.com")

        # Act
        documents = loader.load()

        # Assert
        assert len(documents) == 1
        assert documents[0]["content"] == html_content
        assert documents[0]["metadata"]["source"] == "https://example.com"
        assert documents[0]["metadata"]["content_type"] == "text/plain"


class TestDirectoryLoader:
    """Test cases for the DirectoryLoader class."""

    def test_initialization(self) -> None:
        """Test the initialization of DirectoryLoader."""
        # Arrange & Act
        # Patch Path.exists and Path.is_dir to return True
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                loader = DirectoryLoader(directory_path="./test_dir")

                # Assert
                assert loader.directory_path_str == "./test_dir"
                assert loader.recursive is False  # Default value
                assert loader.glob_pattern == "*.*"  # Default value

    @patch("os.path.isdir", return_value=True)
    @patch("glob.glob")
    def test_load_directory(self, mock_glob, mock_isdir) -> None:
        """Test loading documents from a directory."""
        # Arrange
        mock_glob.return_value = ["file1.txt", "file2.pdf", "file3.csv"]

        # Patch Path.exists and Path.is_dir to return True
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                with (
                    patch("llm_rag.document_processing.loaders.TextFileLoader") as mock_txt_loader,
                    patch("llm_rag.document_processing.loaders.PDFLoader") as mock_pdf_loader,
                    patch("llm_rag.document_processing.loaders.CSVLoader") as mock_csv_loader,
                ):
                    # Setup mock loaders
                    mock_txt_instance = MagicMock()
                    mock_txt_instance.load.return_value = [
                        {"content": "Text content", "metadata": {"source": "file1.txt"}}
                    ]
                    mock_txt_loader.return_value = mock_txt_instance

                    mock_pdf_instance = MagicMock()
                    mock_pdf_instance.load.return_value = [
                        {"content": "PDF content", "metadata": {"source": "file2.pdf"}}
                    ]
                    mock_pdf_loader.return_value = mock_pdf_instance

                    mock_csv_instance = MagicMock()
                    mock_csv_instance.load.return_value = [
                        {"content": "CSV content", "metadata": {"source": "file3.csv"}}
                    ]
                    mock_csv_loader.return_value = mock_csv_instance

                    loader = DirectoryLoader(directory_path="./test_dir")

                    # Act
                    documents = loader.load()

                    # Assert
                    assert len(documents) == 3
                    assert documents[0]["content"] == "Text content"
                    assert documents[1]["content"] == "PDF content"
                    assert documents[2]["content"] == "CSV content"

    @patch("os.path.isdir", return_value=False)
    def test_load_invalid_directory(self, mock_isdir) -> None:
        """Test behavior when directory is invalid."""
        # Arrange
        # Patch Path.exists to return True but Path.is_dir to return False for our test
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=False):
                loader = DirectoryLoader(directory_path="./nonexistent")

                # Act & Assert
                with pytest.raises(NotADirectoryError, match="Directory not found"):
                    loader.load()
