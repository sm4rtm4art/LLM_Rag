"""Unit tests for the document loaders module.

This module contains tests for various document loaders,
including TextFileLoader, PDFLoader, DirectoryLoader, etc.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from langchain.schema import Document

from llm_rag.document_processing.loaders.directory_loader import DirectoryLoader
from llm_rag.document_processing.loaders.file_loaders import CSVLoader, TextFileLoader
from llm_rag.document_processing.loaders.pdf_loaders import PDFLoader

# Import the new loader interfaces and factories - these will be used in additional tests

# Try to import optional loaders
try:
    from llm_rag.document_processing.loaders.json_loader import JSONLoader
except ImportError:
    JSONLoader = None

try:
    from llm_rag.document_processing.loaders.web_loader import WebPageLoader
except ImportError:
    WebPageLoader = None

# Try to import PyMuPDF for PDF tests
try:
    # Using importlib to check if fitz is available without importing it
    import importlib.util

    has_pymupdf = importlib.util.find_spec("fitz") is not None
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
            # Also patch Path.exists to return True to bypass the
            # check in __init__
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
            with patch(
                "src.llm_rag.document_processing.loaders.pdf_loaders.fitz.open", side_effect=Exception("PDF error")
            ):
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


@pytest.mark.skipif(not JSONLoader, reason="JSONLoader not available")
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
                    # The entire JSON structure as one document
                    assert len(documents) == 1
                    assert documents[0]["metadata"]["source"] == "test.json"


@pytest.mark.skipif(not WebPageLoader, reason="WebPageLoader not available")
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
        # Use plain text to avoid BeautifulSoup
        mock_response.headers = {"Content-Type": "text/plain"}
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

    @patch("llm_rag.document_processing.loaders.directory_loader.registry")
    @patch("llm_rag.document_processing.loaders.directory_loader.Path")
    def test_load_directory(self, mock_path_class, mock_registry):
        """Test loading documents from a directory."""
        # Create mock paths
        mock_file1 = MagicMock()
        mock_file1.__str__.return_value = "file1.txt"
        mock_file1.is_file.return_value = True
        mock_file1.name = "file1.txt"
        mock_file1.suffix = ".txt"
        mock_file2 = MagicMock()
        mock_file2.__str__.return_value = "file2.pdf"
        mock_file2.is_file.return_value = True
        mock_file2.name = "file2.pdf"
        mock_file2.suffix = ".pdf"
        mock_file3 = MagicMock()
        mock_file3.__str__.return_value = "file3.csv"
        mock_file3.is_file.return_value = True
        mock_file3.name = "file3.csv"
        mock_file3.suffix = ".csv"

        # Create mock directory path
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mock_dir.is_dir.return_value = True
        mock_dir.glob.return_value = [
            mock_file1,
            mock_file2,
            mock_file3,
        ]
        mock_dir.__str__.return_value = "test_dir"

        # Configure mock path class
        mock_path_class.return_value = mock_dir

        # Create mock loaders
        mock_txt_loader = MagicMock()
        mock_pdf_loader = MagicMock()
        mock_csv_loader = MagicMock()

        # Configure mock registry
        def mock_create_loader_for_file(file_path, **kwargs):
            if str(file_path).endswith(".txt"):
                return mock_txt_loader
            elif str(file_path).endswith(".pdf"):
                return mock_pdf_loader
            elif str(file_path).endswith(".csv"):
                return mock_csv_loader
            return None

        mock_registry.create_loader_for_file.side_effect = mock_create_loader_for_file

        # Configure mock loaders to return documents
        mock_txt_loader.load_from_file.return_value = [Document(page_content="text content")]
        mock_pdf_loader.load_from_file.return_value = [Document(page_content="pdf content")]
        mock_csv_loader.load_from_file.return_value = [Document(page_content="csv content")]

        # Test
        loader = DirectoryLoader("test_dir")
        documents = loader.load()

        # Assert
        assert len(documents) == 3
        assert documents[0].page_content == "text content"
        assert documents[1].page_content == "pdf content"
        assert documents[2].page_content == "csv content"

    def test_load_invalid_directory(self) -> None:
        """Test behavior when directory is not found."""
        # Arrange
        with patch("pathlib.Path.exists", return_value=False):
            loader = DirectoryLoader(directory_path="./nonexistent")

            # Act & Assert
            with pytest.raises(NotADirectoryError):
                loader.load()
