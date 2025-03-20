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

    def __init__(self, web_path: str, **kwargs):
        """Initialize the WebPageLoader.

        Args:
            web_path: URL to fetch content from
            **kwargs: Additional options including:
                - include_images: Whether to include image URLs (default: False)
                - extract_metadata: Whether to extract page metadata (default: True)
                - output_format: Format to return content in ("text", "html", "markdown")

        """
        super().__init__(web_path, **kwargs)
        self.include_images = kwargs.get("include_images", False)
        self.extract_metadata = kwargs.get("extract_metadata", True)
        self.output_format = kwargs.get("output_format", "text")

    def load(self) -> List[Dict[str, Any]]:
        """Load web page with enhanced processing.

        Returns:
            List of documents with content and enhanced metadata

        """
        try:
            import requests

            logger.info(f"Loading web page from: {self.web_path}")
            response = requests.get(self.web_path, timeout=self.timeout, headers=self.headers)
            response.raise_for_status()

            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(response.text, "html.parser")

                # Initialize metadata
                metadata = {
                    "source": self.web_path,
                    "type": "web_page",
                    "status_code": response.status_code,
                    "content_type": response.headers.get("Content-Type", ""),
                }

                # Extract metadata if requested
                if self.extract_metadata:
                    # Title
                    title_tag = soup.find("title")
                    if title_tag:
                        metadata["title"] = title_tag.get_text().strip()

                    # Description
                    description_tag = soup.find("meta", attrs={"name": "description"})
                    if description_tag:
                        metadata["description"] = description_tag.get("content", "")

                    # Keywords
                    keywords_tag = soup.find("meta", attrs={"name": "keywords"})
                    if keywords_tag:
                        metadata["keywords"] = keywords_tag.get("content", "")

                    # Author
                    author_tag = soup.find("meta", attrs={"name": "author"})
                    if author_tag:
                        metadata["author"] = author_tag.get("content", "")

                # Extract image URLs if requested
                if self.include_images:
                    image_urls = []
                    for img in soup.find_all("img"):
                        src = img.get("src")
                        if src:
                            image_urls.append(src)
                    metadata["image_urls"] = image_urls

                # Process the content based on the requested output format
                if self.output_format == "html":
                    # Return the raw HTML
                    content = str(soup)

                elif self.output_format == "markdown":
                    try:
                        # Try to convert HTML to markdown if html2text is available
                        import html2text

                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        h.ignore_images = False
                        content = h.handle(str(soup))
                    except ImportError:
                        # Fall back to plain text
                        logger.warning("html2text not available. Falling back to plain text.")
                        for script in soup(["script", "style"]):
                            script.extract()
                        content = soup.get_text(separator="\n", strip=True)
                        metadata["conversion_note"] = "Markdown conversion failed, using plain text"

                else:  # text format (default)
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    content = soup.get_text(separator=" ", strip=True)

                return [{"content": content, "metadata": metadata}]
            except ImportError:
                # If BeautifulSoup isn't available, use the stub WebLoader
                logger.warning("BeautifulSoup not available. Using basic web loading.")
                return super().load()
        except Exception as e:
            logger.error(f"Error in WebPageLoader for {self.web_path}: {e}")
            return [
                {
                    "content": f"Error retrieving web page: {str(e)}",
                    "metadata": {
                        "source": self.web_path,
                        "type": "web_page",
                        "error": str(e),
                    },
                }
            ]
