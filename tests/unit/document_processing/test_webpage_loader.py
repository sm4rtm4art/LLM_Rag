"""Unit tests for the WebPageLoader class."""

import unittest
from unittest.mock import MagicMock, patch

import pytest

# Try to import WebPageLoader
try:
    from llm_rag.document_processing.loaders import WebPageLoader

    has_webpage_loader = True
except ImportError:
    has_webpage_loader = False

# Skip all tests if WebPageLoader is not available
pytestmark = pytest.mark.skipif(not has_webpage_loader, reason='WebPageLoader not available')


@pytest.mark.skipif(not has_webpage_loader, reason='WebPageLoader not available')
class TestWebPageLoader(unittest.TestCase):
    """Test cases for the WebPageLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_url = 'https://example.com'

    def test_initialization(self):
        """Test the initialization of WebPageLoader."""
        # Arrange & Act
        loader = WebPageLoader(url=self.test_url, headers={'User-Agent': 'Test Agent'}, encoding='latin-1')

        # Assert
        self.assertEqual(loader.url, self.test_url)
        self.assertEqual(loader.headers, {'User-Agent': 'Test Agent'})
        self.assertEqual(loader.encoding, 'latin-1')

    def test_initialization_default_values(self):
        """Test the initialization with default values."""
        # Arrange & Act
        loader = WebPageLoader(url=self.test_url)

        # Assert
        self.assertEqual(loader.url, self.test_url)
        self.assertIn('User-Agent', loader.headers)
        self.assertEqual(loader.encoding, 'utf-8')

    @patch('llm_rag.document_processing.loaders.web_loader.requests.get')
    def test_load_plain_text(self, mock_get):
        """Test loading plain text webpage."""
        # Arrange
        mock_response = MagicMock()
        mock_response.text = 'This is plain text content'
        mock_response.headers = {'Content-Type': 'text/plain'}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Create loader
        loader = WebPageLoader(url=self.test_url)

        with patch.dict('sys.modules', {'bs4': None}):  # No BeautifulSoup
            # Act
            documents = loader.load()

            # Assert
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]['content'], 'This is plain text content')
            self.assertEqual(documents[0]['metadata']['source'], self.test_url)
            self.assertEqual(documents[0]['metadata']['content_type'], 'text/plain')
            self.assertEqual(documents[0]['metadata']['status_code'], 200)

            # Verify request was made correctly
            mock_get.assert_called_once_with(self.test_url, headers=loader.headers, timeout=10, verify=True)

    @patch('llm_rag.document_processing.loaders.web_loader.requests.get')
    def test_load_html_with_bs4(self, mock_get):
        """Test loading HTML with BeautifulSoup parsing."""
        # Arrange
        html_content = '<html><body><h1>Test</h1><p>Content</p></body></html>'
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Mock BeautifulSoup
        mock_bs4 = MagicMock()
        mock_soup = MagicMock()
        mock_soup.get_text.return_value = 'Test Content'
        mock_bs4.BeautifulSoup.return_value = mock_soup

        # Create loader
        loader = WebPageLoader(url=self.test_url)

        with patch.dict('sys.modules', {'bs4': mock_bs4}):
            # Act
            documents = loader.load()

            # Assert
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]['content'], 'Test Content')
            self.assertEqual(documents[0]['metadata']['source'], self.test_url)
            self.assertEqual(documents[0]['metadata']['content_type'], 'text/html')

            # Verify BeautifulSoup was used
            mock_bs4.BeautifulSoup.assert_called_once_with(html_content, 'html.parser')
            mock_soup.get_text.assert_called_once()

    @patch('llm_rag.document_processing.loaders.web_loader.requests.get')
    def test_load_html_without_bs4(self, mock_get):
        """Test loading HTML without BeautifulSoup (falls back to raw HTML)."""
        # Arrange
        html_content = '<html><body><h1>Test</h1><p>Content</p></body></html>'
        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Create loader
        loader = WebPageLoader(url=self.test_url)

        # Mock import error for BeautifulSoup
        with patch.dict('sys.modules', {'bs4': None}):
            # Act
            documents = loader.load()

            # Assert
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]['content'], html_content)  # Raw HTML
            self.assertEqual(documents[0]['metadata']['source'], self.test_url)

    @patch('llm_rag.document_processing.loaders.web_loader.requests.get')
    def test_request_with_custom_headers(self, mock_get):
        """Test making request with custom headers."""
        # Arrange
        mock_response = MagicMock()
        mock_response.text = 'Content'
        mock_response.headers = {'Content-Type': 'text/plain'}
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        custom_headers = {
            'User-Agent': 'Custom Agent',
            'Authorization': 'Bearer token123',
        }

        # Create loader with custom headers
        loader = WebPageLoader(url=self.test_url, headers=custom_headers)

        # Act
        loader.load()

        # Assert
        # Verify custom headers were used
        mock_get.assert_called_once_with(self.test_url, headers=custom_headers, timeout=10, verify=True)

    @patch(
        'llm_rag.document_processing.loaders.web_loader.requests.get',
        side_effect=ConnectionError('Connection error'),
    )
    def test_request_exception(self, mock_get):
        """Test handling of request exceptions."""
        # Arrange
        loader = WebPageLoader(url=self.test_url)

        # Act & Assert
        with self.assertRaises(ConnectionError):
            loader.load()

    @patch('llm_rag.document_processing.loaders.web_loader.requests.get')
    def test_non_200_status_code(self, mock_get):
        """Test handling of non-200 status codes."""
        # Arrange
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = ValueError('404 Not Found')
        mock_get.return_value = mock_response

        # Create loader
        loader = WebPageLoader(url=self.test_url)

        # Act & Assert
        with self.assertRaises(ValueError):
            loader.load()

    @patch(
        'llm_rag.document_processing.loaders.web_loader.requests.get',
        side_effect=ConnectionError('Connection failed'),
    )
    def test_error_handling_connection_error(self, mock_get):
        """Test behavior when connection fails."""
        # Arrange
        loader = WebPageLoader(url='https://example.com')

        # Act & Assert
        with self.assertRaises(ConnectionError):
            loader.load()

    @patch('llm_rag.document_processing.loaders.web_loader.requests.get')
    def test_error_handling_invalid_response(self, mock_get):
        """Test behavior when response is invalid."""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = ValueError('Not found')
        mock_get.return_value = mock_response

        loader = WebPageLoader(url='https://example.com')

        # Act & Assert
        with self.assertRaises(ValueError):
            loader.load()


if __name__ == '__main__':
    unittest.main()
