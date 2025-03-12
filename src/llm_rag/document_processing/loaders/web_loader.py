"""Web document loader.

This module provides a loader for loading documents from web URLs.
"""

import logging
from typing import Dict, Optional
from urllib.parse import urlparse

from ..processors import Documents
from .base import DocumentLoader, registry
from .base import WebLoader as WebLoaderProtocol

logger = logging.getLogger(__name__)

# Optional imports for web processing
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests library not available. Web loading capabilities will be limited.")

try:
    from bs4 import BeautifulSoup

    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    logger.warning("BeautifulSoup not available. HTML processing will be limited.")


class WebLoader(DocumentLoader, WebLoaderProtocol):
    """Load documents from web URLs.

    This loader fetches content from web URLs and processes it into documents.
    It can handle different content types and provides options for extracting
    text from HTML.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        extract_metadata: bool = True,
        extract_images: bool = False,
        encoding: str = "utf-8",
        html_mode: str = "text",  # 'text', 'html', or 'markdown'
        timeout: int = 10,
        verify_ssl: bool = True,
    ):
        """Initialize the web loader.

        Parameters
        ----------
        url : Optional[str], optional
            URL to fetch, by default None
        headers : Optional[Dict[str, str]], optional
            HTTP headers to send with the request, by default None
        extract_metadata : bool, optional
            Whether to extract metadata like title, author, etc., by default True
        extract_images : bool, optional
            Whether to extract image URLs from HTML, by default False
        encoding : str, optional
            Text encoding to use, by default "utf-8"
        html_mode : str, optional
            How to process HTML content ('text', 'html', or 'markdown'), by default "text"
        timeout : int, optional
            Request timeout in seconds, by default 10
        verify_ssl : bool, optional
            Whether to verify SSL certificates, by default True

        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests library is required for WebLoader. Install it with 'pip install requests'")

        self.url = url
        self.headers = headers or {}
        # Add default User-Agent if not provided
        if "User-Agent" not in self.headers:
            self.headers["User-Agent"] = "Mozilla/5.0 (compatible; LLM-RAG/1.0; +https://github.com/example/llm-rag)"
        self.extract_metadata = extract_metadata
        self.extract_images = extract_images
        self.encoding = encoding
        self.html_mode = html_mode
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        # Validate HTML mode
        if html_mode not in ["text", "html", "markdown"]:
            raise ValueError("Invalid HTML mode. Must be one of: 'text', 'html', 'markdown'")

        # Check if we have markdown support if needed
        if html_mode == "markdown":
            try:
                import importlib.util

                if importlib.util.find_spec("html2text") is not None:
                    self._has_html2text = True
                else:
                    raise ImportError("html2text module not found")
            except ImportError:
                logger.warning("html2text not available. Using plain text extraction instead.")
                self._has_html2text = False
                self.html_mode = "text"
        else:
            self._has_html2text = False

    def load(self) -> Documents:
        """Load documents from the URL specified during initialization.

        Returns
        -------
        Documents
            List of documents loaded from the URL.

        Raises
        ------
        ValueError
            If URL was not provided during initialization.

        """
        if not self.url:
            raise ValueError("No URL provided. Either initialize with a URL or use load_from_url.")

        return self.load_from_url(self.url, self.headers)

    def load_from_url(self, url: str, headers: Optional[Dict[str, str]] = None) -> Documents:
        """Load documents from a URL.

        Parameters
        ----------
        url : str
            URL to fetch.
        headers : Optional[Dict[str, str]], optional
            HTTP headers to send with the request, by default None

        Returns
        -------
        Documents
            List of documents loaded from the URL.

        Raises
        ------
        ConnectionError
            If there's a network connection error.
        ValueError
            If there's an HTTP error (like 404, 500, etc.)
        Exception
            For other unexpected errors.

        """
        headers = headers or self.headers

        try:
            # Fetch content from URL
            response = requests.get(url, headers=headers, timeout=self.timeout, verify=self.verify_ssl)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Determine content type
            content_type = response.headers.get("Content-Type", "").lower()

            # Create base metadata
            metadata = {
                "source": url,
                "content_type": content_type,
                "status_code": response.status_code,
            }

            # Extract domain for metadata
            parsed_url = urlparse(url)
            metadata["domain"] = parsed_url.netloc

            # Process content based on content type
            if "text/html" in content_type and BEAUTIFULSOUP_AVAILABLE:
                documents = self._process_html(response.text, url, metadata)
            else:
                # Process as plain text
                documents = [{"content": response.text, "metadata": metadata}]

            return documents

        except requests.exceptions.ConnectionError as e:
            # Network connection error
            logger.error(f"Connection error loading URL {url}: {e}")
            raise

        except requests.exceptions.HTTPError as e:
            # HTTP error (404, 500, etc.)
            logger.error(f"HTTP error loading URL {url}: {e}")
            raise ValueError(str(e)) from e

        except Exception as e:
            # Other unexpected errors
            logger.error(f"Error loading URL {url}: {e}")
            raise

    def _process_html(self, html_content: str, url: str, metadata: Dict) -> Documents:
        """Process HTML content into documents.

        Parameters
        ----------
        html_content : str
            HTML content to process.
        url : str
            Source URL.
        metadata : Dict
            Base metadata to include.

        Returns
        -------
        Documents
            List of documents extracted from the HTML.

        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract metadata if requested
        if self.extract_metadata:
            # Title
            title_tag = soup.find("title")
            if title_tag and title_tag.text:
                metadata["title"] = title_tag.text.strip()

            # Description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                metadata["description"] = meta_desc["content"].strip()

            # Author
            meta_author = soup.find("meta", attrs={"name": "author"})
            if meta_author and meta_author.get("content"):
                metadata["author"] = meta_author["content"].strip()

        # Extract text based on chosen mode
        if self.html_mode == "html":
            # Return HTML as-is
            content = html_content
        elif self.html_mode == "markdown" and self._has_html2text:
            # Convert to markdown using html2text
            import html2text

            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = not self.extract_images
            content = h.handle(html_content)
        else:
            # Extract just the text
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Get text and clean it
            content = soup.get_text(separator="\n")
            # Clean up whitespace
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = "\n".join(chunk for chunk in chunks if chunk)

        # Extract image URLs if requested
        if self.extract_images and "text/html" in metadata.get("content_type", ""):
            image_urls = [img.get("src") for img in soup.find_all("img") if img.get("src")]
            if image_urls:
                metadata["image_urls"] = image_urls

        return [{"content": content, "metadata": metadata}]


# Register the loader
registry.register(WebLoader, extensions=["html", "htm"])

# Alias for backward compatibility
WebPageLoader = WebLoader
registry.register(WebPageLoader, name="WebPageLoader")
