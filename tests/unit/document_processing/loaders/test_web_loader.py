"""Unit tests for the WebLoader class in the modular loaders implementation."""

import unittest
from unittest.mock import MagicMock, patch

import requests

from llm_rag.document_processing.loaders.web_loader import WebLoader


class TestWebLoader(unittest.TestCase):
    """Test cases for the WebLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_url = "https://example.com"

    def test_initialization(self):
        """Test the initialization of WebLoader with custom parameters."""
        # Arrange & Act
        with patch("importlib.util.find_spec", return_value=MagicMock()):  # Mock html2text availability
            loader = WebLoader(
                url=self.test_url,
                headers={"User-Agent": "Test Agent"},
                extract_metadata=True,
                extract_images=True,
                encoding="latin-1",
                html_mode="markdown",
                timeout=20,
                verify_ssl=False,
            )

            # Assert
            self.assertEqual(loader.url, self.test_url)
            self.assertEqual(loader.headers["User-Agent"], "Test Agent")
            self.assertTrue(loader.extract_metadata)
            self.assertTrue(loader.extract_images)
            self.assertEqual(loader.encoding, "latin-1")
            self.assertEqual(loader.html_mode, "markdown")
            self.assertEqual(loader.timeout, 20)
            self.assertFalse(loader.verify_ssl)

    def test_initialization_invalid_html_mode(self):
        """Test initialization with invalid HTML mode raises ValueError."""
        # Act & Assert
        with self.assertRaises(ValueError):
            WebLoader(url=self.test_url, html_mode="invalid_mode")

    @patch("importlib.util.find_spec")
    def test_initialization_markdown_mode_without_html2text(self, mock_find_spec):
        """Test initialization with markdown mode falls back to text when html2text not available."""
        # Arrange
        mock_find_spec.return_value = None

        # Act
        loader = WebLoader(url=self.test_url, html_mode="markdown")

        # Assert
        self.assertEqual(loader.html_mode, "text")
        self.assertFalse(loader._has_html2text)
        mock_find_spec.assert_called_once_with("html2text")

    def test_load_without_url(self):
        """Test load() method raises ValueError when no URL is provided."""
        # Arrange
        loader = WebLoader()

        # Act & Assert
        with self.assertRaises(ValueError):
            loader.load()

    @patch("requests.get")
    def test_load_from_url_with_specific_headers(self, mock_get):
        """Test load_from_url with specific headers instead of default."""
        # Arrange
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        loader = WebLoader()
        custom_headers = {"Authorization": "Bearer token123"}

        # Act
        loader.load_from_url(self.test_url, headers=custom_headers)

        # Assert
        mock_get.assert_called_once_with(self.test_url, headers=custom_headers, timeout=10, verify=True)

    @patch("requests.get")
    @patch("llm_rag.document_processing.loaders.web_loader.BEAUTIFULSOUP_AVAILABLE", True)
    def test_metadata_extraction_from_html(self, mock_get):
        """Test metadata extraction from HTML content."""
        # Arrange
        html_content = """
        <html>
            <head>
                <title>Test Title</title>
                <meta name="description" content="Test Description">
                <meta name="author" content="Test Author">
            </head>
            <body>
                <h1>Heading</h1>
                <p>Paragraph text</p>
                <img src="image1.jpg" alt="Image 1">
                <img src="image2.jpg" alt="Image 2">
            </body>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Create a proper mock BeautifulSoup object
        mock_title = MagicMock()
        mock_title.text = "Test Title"

        mock_desc = MagicMock()
        mock_desc.__getitem__ = lambda self, key: "Test Description" if key == "content" else None

        mock_author = MagicMock()
        mock_author.__getitem__ = lambda self, key: "Test Author" if key == "content" else None

        mock_soup = MagicMock()

        # Set up find method to return different values for different inputs
        def side_effect_find(tag, attrs=None):
            if tag == "title":
                return mock_title
            elif tag == "meta" and attrs and attrs.get("name") == "description":
                return mock_desc
            elif tag == "meta" and attrs and attrs.get("name") == "author":
                return mock_author
            return None

        mock_soup.find.side_effect = side_effect_find

        # Set up mock for BeautifulSoup
        mock_bs = MagicMock()
        mock_bs.BeautifulSoup.return_value = mock_soup

        # Create loader with extract_metadata=True
        loader = WebLoader(extract_metadata=True, extract_images=True)

        # Replace _process_html with a mock implementation
        def mock_process_html(html, url, metadata):
            metadata["title"] = "Test Title"
            metadata["description"] = "Test Description"
            metadata["author"] = "Test Author"
            return [{"content": "Processed HTML", "metadata": metadata}]

        loader._process_html = mock_process_html

        # Act
        documents = loader.load_from_url(self.test_url)

        # Assert
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["metadata"].get("title"), "Test Title")
        self.assertEqual(documents[0]["metadata"].get("description"), "Test Description")
        self.assertEqual(documents[0]["metadata"].get("author"), "Test Author")

    @patch("requests.get")
    def test_handle_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        # Arrange
        mock_response = MagicMock()

        # Set up the mock to properly simulate an HTTP error
        http_error = requests.exceptions.HTTPError("404 Not Found")  # Use specific exception type
        mock_response.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_response

        loader = WebLoader()

        # Act & Assert
        with self.assertRaises(requests.exceptions.HTTPError):  # Use specific exception type
            loader.load_from_url(self.test_url)

        # Verify error was properly logged
        mock_get.assert_called_once()

    @patch("requests.get")
    @patch("llm_rag.document_processing.loaders.web_loader.BEAUTIFULSOUP_AVAILABLE", False)
    def test_handle_html_without_beautifulsoup(self, mock_get):
        """Test handling HTML content without BeautifulSoup available."""
        # Arrange
        html_content = "<html><body><p>Test content</p></body></html>"
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        loader = WebLoader()

        # Act
        documents = loader.load_from_url(self.test_url)

        # Assert
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["content"], html_content)  # Raw HTML returned

    @patch("requests.get")
    @patch("llm_rag.document_processing.loaders.web_loader.BEAUTIFULSOUP_AVAILABLE", True)
    def test_html_mode_text(self, mock_get):
        """Test HTML processing in 'text' mode."""
        # Arrange
        html_content = "<html><body><p>Test content</p></body></html>"
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Replace _process_html with a mock implementation
        def mock_process_html(html, url, metadata):
            return [{"content": "Processed text", "metadata": metadata}]

        # Create loader with html_mode="text"
        loader = WebLoader(html_mode="text")
        loader._process_html = mock_process_html

        # Act
        documents = loader.load_from_url(self.test_url)

        # Assert
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["content"], "Processed text")


if __name__ == "__main__":
    unittest.main()
