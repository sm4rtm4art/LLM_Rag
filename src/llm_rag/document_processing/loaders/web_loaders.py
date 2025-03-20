"""Web-based document loaders.

This module provides components for loading documents from web URLs,
with support for HTML parsing and text extraction.
"""

from typing import Any, Dict, List

from llm_rag.utils.logging import get_logger

from .base import DocumentLoader

logger = get_logger(__name__)


class WebLoader(DocumentLoader):
    """Load documents from web URLs."""

    def __init__(self, web_path: str, **kwargs):
        """Initialize the WebLoader.

        Args:
            web_path: URL to fetch content from
            **kwargs: Additional options like headers, timeout, etc.

        """
        self.web_path = web_path
        self.kwargs = kwargs
        self.timeout = kwargs.get("timeout", 10)
        self.headers = kwargs.get("headers", {})

    def load(self) -> List[Dict[str, Any]]:
        """Load content from web URL.

        Returns:
            List of documents with content and metadata

        """
        try:
            import requests

            logger.info(f"Loading web content from: {self.web_path}")
            response = requests.get(self.web_path, timeout=self.timeout, headers=self.headers)
            response.raise_for_status()

            # Try to parse HTML if possible
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "html.parser")

                # Try to extract title if available
                title = None
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text().strip()

                # Remove script and style elements that contain JavaScript/CSS
                for script in soup(["script", "style"]):
                    script.extract()

                # Get the text content
                text = soup.get_text(separator=" ", strip=True)

                metadata = {
                    "source": self.web_path,
                    "type": "web",
                    "status_code": response.status_code,
                }

                if title:
                    metadata["title"] = title

                return [{"content": text, "metadata": metadata}]
            except ImportError:
                # If BeautifulSoup isn't available, just use the raw HTML
                logger.warning("BeautifulSoup not available. Using raw HTML.")
                return [
                    {
                        "content": response.text,
                        "metadata": {
                            "source": self.web_path,
                            "type": "web_raw",
                            "status_code": response.status_code,
                        },
                    }
                ]
        except ImportError:
            logger.warning("Requests library not available. Web content cannot be retrieved.")
            return [
                {
                    "content": f"Web content unavailable. URL: {self.web_path}",
                    "metadata": {
                        "source": self.web_path,
                        "type": "web",
                        "error": "Web libraries not available",
                    },
                }
            ]
        except Exception as e:
            logger.error(f"Error loading web content from {self.web_path}: {e}")
            return [
                {
                    "content": f"Error retrieving web content: {str(e)}",
                    "metadata": {
                        "source": self.web_path,
                        "type": "web",
                        "error": str(e),
                    },
                }
            ]


class WebPageLoader(WebLoader):
    """Enhanced loader for web pages with additional capabilities."""

    def __init__(self, web_path=None, url=None, headers=None, encoding="utf-8", **kwargs):
        """Initialize the WebPageLoader.

        Args:
            web_path: URL to fetch content from (default=None)
            url: Alternative name for web_path (default=None)
            headers: HTTP headers to include in request (default=None)
            encoding: Text encoding to use (default="utf-8")
            **kwargs: Additional options including:
                - include_images: Whether to include image URLs (default: False)
                - extract_metadata: Whether to extract page metadata (default: True)
                - output_format: Format to return content in ("text", "html", "markdown")

        Raises:
            ValueError: If neither web_path nor url is provided

        """
        # Check if url is provided instead of web_path
        if web_path is None and url is not None:
            web_path = url
        elif web_path is None and url is None:
            raise ValueError("Either web_path or url must be provided")

        # Set up default headers if not provided
        if headers is None:
            headers = {"User-Agent": "LLM-RAG WebPageLoader/1.0"}

        # Pass headers to parent
        super().__init__(web_path, headers=headers, **kwargs)

        self.include_images = kwargs.get("include_images", False)
        self.extract_metadata = kwargs.get("extract_metadata", True)
        self.output_format = kwargs.get("output_format", "text")

        # For compatibility with tests
        self.url = web_path
        self.encoding = encoding

    def load(self) -> List[Dict[str, Any]]:
        """Load web page with enhanced processing.

        Returns:
            List of documents with content and enhanced metadata

        Raises:
            ConnectionError: If connection to the URL fails
            ValueError: If response status is invalid

        """
        try:
            import requests

            logger.info(f"Loading web page from: {self.web_path}")
            response = requests.get(self.web_path, timeout=self.timeout, headers=self.headers, verify=True)

            # This will raise ValueError for non-200 status
            response.raise_for_status()

            # Process with BeautifulSoup if available
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "html.parser")

                # Extract page title
                title = None
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()

                # Extract page content based on output format
                if self.output_format == "html":
                    # Return HTML content
                    content = str(soup)
                elif self.output_format == "markdown":
                    # Convert to markdown if possible
                    try:
                        import html2text

                        h2t = html2text.HTML2Text()
                        content = h2t.handle(str(soup))
                    except ImportError:
                        logger.warning("html2text not available. Using plain text.")
                        content = soup.get_text(separator="\n", strip=True)
                else:
                    # Default to plain text
                    content = soup.get_text(separator="\n", strip=True)

                # Extract metadata
                metadata = {
                    "source": self.web_path,
                    "title": title,
                    "content_type": response.headers.get("Content-Type", "text/html"),
                    "status_code": response.status_code,
                }

                return [{"content": content, "metadata": metadata}]

            except ImportError:
                logger.warning("BeautifulSoup not available. Using raw HTML.")

                # Process without BeautifulSoup
                content = response.text

                metadata = {
                    "source": self.web_path,
                    "content_type": response.headers.get("Content-Type", "text/html"),
                    "status_code": response.status_code,
                }

                return [{"content": content, "metadata": metadata}]

        except (ConnectionError, ValueError) as e:
            # Re-raise these specific errors for the tests to catch
            logger.error(f"Error in WebPageLoader for {self.web_path}: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Error in WebPageLoader for {self.web_path}: {str(e)}")
            return [
                {
                    "content": f"Error loading content from {self.web_path}: {str(e)}",
                    "metadata": {"source": self.web_path, "error": str(e)},
                }
            ]
