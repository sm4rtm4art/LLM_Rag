"""Unit tests for the document loaders module.

This module contains tests for various document loaders,
including TextFileLoader, PDFLoader, DirectoryLoader, etc.
"""

from unittest.mock import MagicMock, mock_open, patch
from pathlib import Path

import pytest

# Import required legacy loaders
from llm_rag.document_processing import (
    CSVLoader,
    DirectoryLoader,
    PDFLoader,
    TextFileLoader,
)

# Import the new loader interfaces and factories - these will be used in additional tests

# Try to import optional loaders
try:
    from llm_rag.document_processing import JSONLoader

    has_json_loader = True
except ImportError:
    has_json_loader = False

try:
    from llm_rag.document_processing import WebPageLoader

    has_webpage_loader = True
except ImportError:
    has_webpage_loader = False

# Try to import PyMuPDF for PDF tests
try:
    import fitz

    has_pymupdf = True
except ImportError:
    has_pymupdf = False

# EnhancedPDFLoader is not used in this test file, so we don't need to import it
# Just define the flag for skipping tests
has_enhanced_pdf_loader = False


@pytest.mark.refactoring
class TestTextFileLoader:
    """Test cases for the TextFileLoader class."""

    def test_initialization(self) -> None:
        """Test the initialization of TextFileLoader."""
        # Arrange & Act
        loader = TextFileLoader(file_path="test.txt")

        # Assert
        assert loader.file_path == Path("test.txt")

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


@pytest.mark.refactoring
class TestPDFLoader:
    """Test cases for the PDFLoader class."""

    def test_initialization(self) -> None:
        """Test the initialization of PDFLoader."""
        # Arrange & Act
        loader = PDFLoader(file_path="test.pdf")

        # Assert
        assert loader.file_path == Path("test.pdf")
        assert loader.extract_images is False  # Default value
        assert loader.extract_tables is False  # Default value

    def test_load_valid_pdf(self) -> None:
        """Test loading content from a valid PDF file."""
        # Check if fitz is available
        if not has_pymupdf:
            pytest.skip("PyMuPDF (fitz) not available")

        # Arrange
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.llm_rag.document_processing.loaders.pdf_loaders.fitz.open") as mock_fitz_open:
                mock_pdf = MagicMock()
                mock_page1 = MagicMock()
                mock_page1.get_text.return_value = "Page 1 content"
                mock_page2 = MagicMock()
                mock_page2.get_text.return_value = "Page 2 content"

                # Configure the mock to return multiple pages
                mock_pdf.__len__.return_value = 2
                mock_pdf.__getitem__.side_effect = [mock_page1, mock_page2]
                # Set up the context manager
                mock_fitz_open.return_value.__enter__.return_value = mock_pdf

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

    def test_load_invalid_pdf(self) -> None:
        """Test behavior when PDF loading fails."""
        # Check if fitz is available
        if not has_pymupdf:
            pytest.skip("PyMuPDF (fitz) not available")

        # Arrange
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.llm_rag.document_processing.loaders.pdf_loaders.fitz.open", side_effect=Exception("PDF error")):
                loader = PDFLoader(file_path="test.pdf")

                # Act & Assert
                with pytest.raises(Exception, match="Error loading PDF file"):
                    loader.load()


@pytest.mark.refactoring
class TestCSVLoader:
    """Test cases for the CSVLoader class."""

    def test_initialization(self) -> None:
        """Test the initialization of CSVLoader."""
        # Arrange & Act
        loader = CSVLoader(file_path="test.csv")

        # Assert
        assert loader.file_path == Path("test.csv")
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
                    assert "name: John" in documents[0]["content"]
                    assert "age: 30" in documents[0]["content"]
                    assert documents[0]["metadata"]["filetype"] == "csv"

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
                    documents = loader.load()

                    # Assert
                    assert len(documents) == 2
                    assert "name: John" in documents[0]["content"]
                    assert "age: 30" in documents[0]["content"]


@pytest.mark.skipif(not has_json_loader, reason="JSONLoader not available")
@pytest.mark.refactoring
class TestJSONLoader:
    """Test cases for the JSONLoader class."""

    def test_initialization(self) -> None:
        """Test the initialization of JSONLoader."""
        # Arrange & Act
        loader = JSONLoader(file_path="test.json")

        # Assert
        assert loader.file_path == Path("test.json")
        assert loader.content_key is None  # Default value
        assert loader.metadata_keys == []  # Default value

    def test_load_valid_json(self) -> None:
        """Test loading content from a valid JSON file."""
        # Arrange
        json_content = '{"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]}'
        mock_file = mock_open(read_data=json_content)

        with patch("builtins.open", mock_file):
            with patch("json.load") as mock_json_load:
                mock_json_load.return_value = {"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]}
                
                # Also patch Path.exists to return True
                with patch("pathlib.Path.exists", return_value=True):
                    loader = JSONLoader(file_path="test.json", content_key="name")

                    # Act
                    documents = loader.load()

                    # Assert
                    mock_file.assert_called_once_with(Path("test.json"), "r", encoding="utf-8")
                    assert len(documents) == 1  # The entire JSON structure as one document
                    assert documents[0]["metadata"]["source"] == "test.json"


@pytest.mark.skipif(not has_webpage_loader, reason="WebPageLoader not available")
@pytest.mark.refactoring
class TestWebPageLoader:
    """Test cases for the WebPageLoader class."""

    def test_initialization(self) -> None:
        """Test the initialization of WebPageLoader."""
        # Arrange & Act
        loader = WebPageLoader(url="https://example.com")

        # Assert
        assert loader.url == "https://example.com"
        assert loader.encoding == "utf-8"  # Default encoding

    @patch("llm_rag.document_processing.loaders.web_loader.requests.get")
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


@pytest.mark.refactoring
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
                assert loader.directory_path == Path("./test_dir")
                assert loader.recursive is False  # Default value
                assert loader.glob_pattern == "*.*"  # Default value

    @patch("os.path.isdir", return_value=True)
    @patch("glob.glob")
    def test_load_directory(self, mock_glob, mock_isdir) -> None:
        """Test loading documents from a directory."""
        # Arrange
        mock_file_paths = [Path("file1.txt"), Path("file2.pdf"), Path("file3.csv")]
        
        # Patch Path.exists and Path.is_dir to return True
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                # Mock the directory_path.glob method to return our mock files
                with patch("pathlib.Path.glob", return_value=mock_file_paths):
                    # Mock the file.is_file method to return True
                    with patch("pathlib.Path.is_file", return_value=True):
                        # Mock the load_document function
                        with patch("src.llm_rag.document_processing.loaders.directory_loader.load_document") as mock_load_document:
                            # Setup mock documents to be returned
                            mock_load_document.side_effect = [
                                [{"content": "Text content", "metadata": {"source": "file1.txt"}}],
                                [{"content": "PDF content", "metadata": {"source": "file2.pdf"}}],
                                [{"content": "CSV content", "metadata": {"source": "file3.csv"}}],
                            ]

                            loader = DirectoryLoader(directory_path="./test_dir")

                            # Act
                            documents = loader.load()

                            # Assert
                            assert len(documents) == 3
                            assert documents[0]["content"] == "Text content"
                            assert documents[1]["content"] == "PDF content"
                            assert documents[2]["content"] == "CSV content"

    def test_load_invalid_directory(self) -> None:
        """Test behavior when directory is not found."""
        # Arrange
        with patch("pathlib.Path.exists", return_value=False):
            loader = DirectoryLoader(directory_path="./nonexistent")

            # Act & Assert
            with pytest.raises(NotADirectoryError):
                loader.load()
