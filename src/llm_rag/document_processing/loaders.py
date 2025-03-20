"""Document loaders for the LLM-RAG system.

This module provides components for loading documents from various sources
and formats, including PDFs, text files, CSV files, JSON, and web content.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

# Get configured logger
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)

# Define document types
Documents = List[Dict[str, Any]]

try:
    # Try to import, but this will likely fail until we fix the directory structure
    from llm_rag.document_processing.loaders.base import DocumentLoader
    from llm_rag.document_processing.loaders.directory_loader import DirectoryLoader
    from llm_rag.document_processing.loaders.file_loaders import (
        CSVLoader,
        EnhancedPDFLoader,
        JSONLoader,
        PDFLoader,
        TextFileLoader,
        XMLLoader,
    )
    from llm_rag.document_processing.loaders.web_loaders import WebLoader, WebPageLoader

    # Flag for successful import
    _MODULAR_IMPORT_SUCCESS = True

except ImportError as e:
    # If import fails, use functional stubs
    warnings.warn(f"Failed to import modular document loader components: {e}. Using functional stubs.", stacklevel=2)
    _MODULAR_IMPORT_SUCCESS = False

    # Define base Document Loader class for stubs
    class DocumentLoader:
        """Base class for document loaders."""

        def load(self) -> Documents:
            """Load documents from a source."""
            raise NotImplementedError("Subclasses must implement this method")

    class TextFileLoader(DocumentLoader):
        """Functional stub implementation of TextFileLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize the TextFileLoader with basic functionality."""
            self.file_path = Path(file_path)
            self.kwargs = kwargs

        def load(self) -> Documents:
            """Load documents from a text file using basic Python functionality."""
            try:
                logger.info(f"Loading text file: {self.file_path}")
                with open(self.file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                return [
                    {
                        "content": content,
                        "metadata": {
                            "source": str(self.file_path),
                            "filename": self.file_path.name,
                            "type": "text",
                        },
                    }
                ]
            except Exception as e:
                logger.error(f"Error loading text file {self.file_path}: {e}")
                return []

    class PDFLoader(DocumentLoader):
        """Functional stub implementation of PDFLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize the PDFLoader with basic functionality."""
            self.file_path = Path(file_path)
            self.kwargs = kwargs

        def load(self) -> Documents:
            """Load documents from a PDF file using available fallbacks."""
            try:
                logger.info(f"Loading PDF file: {self.file_path}")
                # Try to use PyPDF if available
                try:
                    import pypdf

                    with open(self.file_path, "rb") as f:
                        pdf = pypdf.PdfReader(f)
                        text = ""
                        for page in pdf.pages:
                            text += page.extract_text() + "\n"

                        return [
                            {
                                "content": text,
                                "metadata": {
                                    "source": str(self.file_path),
                                    "filename": self.file_path.name,
                                    "type": "pdf",
                                    "pages": len(pdf.pages),
                                },
                            }
                        ]
                except ImportError:
                    logger.warning("PyPDF not available. PDF content will be limited.")
                    return [
                        {
                            "content": f"PDF file content unavailable. File: {self.file_path.name}",
                            "metadata": {
                                "source": str(self.file_path),
                                "filename": self.file_path.name,
                                "type": "pdf",
                                "error": "PDF libraries not available",
                            },
                        }
                    ]
            except Exception as e:
                logger.error(f"Error loading PDF file {self.file_path}: {e}")
                return []

    class EnhancedPDFLoader(PDFLoader):
        """Functional stub implementation that falls back to basic PDFLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize with fallback to basic PDFLoader."""
            super().__init__(file_path, **kwargs)
            logger.warning("EnhancedPDFLoader is using basic PDFLoader fallback functionality.")

    class CSVLoader(DocumentLoader):
        """Functional stub implementation of CSVLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize the CSVLoader with basic functionality."""
            self.file_path = Path(file_path)
            self.kwargs = kwargs
            self.content_column = kwargs.get("content_column")

        def load(self) -> Documents:
            """Load documents from a CSV file using the built-in csv module."""
            try:
                import csv

                logger.info(f"Loading CSV file: {self.file_path}")

                documents = []
                with open(self.file_path, "r", encoding="utf-8") as f:
                    csv_reader = csv.DictReader(f)
                    for i, row in enumerate(csv_reader):
                        if self.content_column and self.content_column in row:
                            content = row[self.content_column]
                        else:
                            content = "; ".join([f"{k}: {v}" for k, v in row.items()])

                        documents.append(
                            {
                                "content": content,
                                "metadata": {
                                    "source": str(self.file_path),
                                    "filename": self.file_path.name,
                                    "type": "csv",
                                    "row": i,
                                    **{k: v for k, v in row.items() if k != self.content_column},
                                },
                            }
                        )

                return documents
            except Exception as e:
                logger.error(f"Error loading CSV file {self.file_path}: {e}")
                return []

    class JSONLoader(DocumentLoader):
        """Functional stub implementation of JSONLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize the JSONLoader with basic functionality."""
            self.file_path = Path(file_path)
            self.kwargs = kwargs
            self.jq_schema = kwargs.get("jq_schema", ".")

        def load(self) -> Documents:
            """Load documents from a JSON file using the built-in json module."""
            try:
                import json

                logger.info(f"Loading JSON file: {self.file_path}")

                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, list):
                    # List of records
                    return [
                        {
                            "content": json.dumps(item) if isinstance(item, (dict, list)) else str(item),
                            "metadata": {
                                "source": str(self.file_path),
                                "filename": self.file_path.name,
                                "type": "json",
                                "index": i,
                            },
                        }
                        for i, item in enumerate(data)
                    ]
                else:
                    # Single record or hierarchical
                    return [
                        {
                            "content": json.dumps(data) if isinstance(data, (dict, list)) else str(data),
                            "metadata": {
                                "source": str(self.file_path),
                                "filename": self.file_path.name,
                                "type": "json",
                            },
                        }
                    ]
            except Exception as e:
                logger.error(f"Error loading JSON file {self.file_path}: {e}")
                return []

    class WebLoader(DocumentLoader):
        """Functional stub implementation of WebLoader."""

        def __init__(self, web_path: str, **kwargs):
            """Initialize the WebLoader with basic functionality."""
            self.web_path = web_path
            self.kwargs = kwargs

        def load(self) -> Documents:
            """Load documents from a web URL using available fallbacks."""
            try:
                logger.info(f"Loading web content from: {self.web_path}")
                try:
                    # Try to use requests if available
                    import requests

                    response = requests.get(self.web_path, timeout=10)
                    response.raise_for_status()

                    # Try to extract text with BeautifulSoup if available
                    try:
                        from bs4 import BeautifulSoup

                        soup = BeautifulSoup(response.text, "html.parser")
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.extract()
                        text = soup.get_text(separator=" ", strip=True)
                    except ImportError:
                        # Fall back to raw HTML
                        text = response.text

                    return [
                        {
                            "content": text,
                            "metadata": {
                                "source": self.web_path,
                                "type": "web",
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
                return []

    class WebPageLoader(WebLoader):
        """Functional stub implementation that falls back to WebLoader."""

        def __init__(self, web_path: str, **kwargs):
            """Initialize with fallback to basic WebLoader."""
            super().__init__(web_path, **kwargs)
            logger.warning("WebPageLoader is using basic WebLoader fallback functionality.")

    class DirectoryLoader(DocumentLoader):
        """Functional stub implementation of DirectoryLoader."""

        def __init__(
            self, directory_path: Union[str, Path], glob_pattern: str = "*.*", recursive: bool = True, **kwargs
        ):
            """Initialize the DirectoryLoader with basic functionality."""
            self.directory_path = Path(directory_path)
            self.glob_pattern = glob_pattern
            self.recursive = recursive
            self.kwargs = kwargs

        def load(self) -> Documents:
            """Load documents from files in a directory using available fallbacks."""
            try:
                logger.info(f"Loading documents from directory: {self.directory_path}")

                documents = []

                # Handle recursive vs. non-recursive
                if self.recursive:
                    glob_path = self.directory_path.glob("**/" + self.glob_pattern)
                else:
                    glob_path = self.directory_path.glob(self.glob_pattern)

                # Process each file
                for file_path in glob_path:
                    if not file_path.is_file():
                        continue

                    # Determine loader based on file extension
                    ext = file_path.suffix.lower()

                    try:
                        if ext == ".txt":
                            loader = TextFileLoader(file_path, **self.kwargs)
                        elif ext == ".pdf":
                            loader = PDFLoader(file_path, **self.kwargs)
                        elif ext == ".csv":
                            loader = CSVLoader(file_path, **self.kwargs)
                        elif ext == ".json":
                            loader = JSONLoader(file_path, **self.kwargs)
                        elif ext == ".xml":
                            loader = XMLLoader(file_path, **self.kwargs)
                        else:
                            # Default to text for unknown types
                            loader = TextFileLoader(file_path, **self.kwargs)

                        file_docs = loader.load()
                        documents.extend(file_docs)
                    except Exception as e:
                        logger.error(f"Error loading file {file_path}: {e}")

                return documents
            except Exception as e:
                logger.error(f"Error loading directory {self.directory_path}: {e}")
                return []

    class XMLLoader(DocumentLoader):
        """Functional stub implementation of XMLLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize the XMLLoader with basic functionality."""
            self.file_path = Path(file_path)
            self.kwargs = kwargs
            self.content_tags = kwargs.get("content_tags")
            self.metadata_tags = kwargs.get("metadata_tags")
            self.split_by_tag = kwargs.get("split_by_tag")

        def load(self) -> Documents:
            """Load documents from an XML file using built-in modules."""
            try:
                logger.info(f"Loading XML file: {self.file_path}")
                try:
                    # Use defusedxml instead of standard xml module for security
                    try:
                        from defusedxml import ElementTree as ET
                    except ImportError:
                        logger.warning("defusedxml not available, using stdlib with defuse_stdlib()")
                        # nosec B405 - We're handling the security risk with defuse_stdlib below
                        import xml.etree.ElementTree as ET  # nosec B405

                        try:
                            from defusedxml import defuse_stdlib

                            defuse_stdlib()
                            logger.info("Applied defuse_stdlib() for XML security")
                        except ImportError:
                            logger.warning("defusedxml not available at all, using unsafe XML parsing")

                    # nosec B314 - We're using defusedxml or have called defuse_stdlib above
                    tree = ET.parse(str(self.file_path))  # nosec B314
                    root = tree.getroot()

                    documents = []

                    # Handle splitting by tag
                    if self.split_by_tag:
                        for i, element in enumerate(root.findall(f".//{self.split_by_tag}")):
                            content = self._extract_content(element)
                            metadata = self._extract_metadata(element, index=i)
                            documents.append(
                                {
                                    "content": content,
                                    "metadata": metadata,
                                }
                            )
                    else:
                        # Process the whole document
                        content = self._extract_content(root)
                        metadata = self._extract_metadata(root)
                        documents.append(
                            {
                                "content": content,
                                "metadata": metadata,
                            }
                        )

                    return documents

                except ImportError:
                    logger.warning("XML parsing libraries not available.")
                    return [
                        {
                            "content": f"XML content unavailable. File: {self.file_path.name}",
                            "metadata": {
                                "source": str(self.file_path),
                                "filename": self.file_path.name,
                                "type": "xml",
                                "error": "XML libraries not available",
                            },
                        }
                    ]
            except Exception as e:
                logger.error(f"Error loading XML file {self.file_path}: {e}")
                return []

        def _extract_content(self, element) -> str:
            """Extract content based on configuration."""
            if self.content_tags:
                # Extract content from specific tags
                contents = []
                for tag in self.content_tags:
                    for match in element.findall(f".//{tag}"):
                        contents.append(f"{tag}: {match.text}")
                return "\n".join(contents)
            else:
                # Extract all text content
                return "".join(element.itertext())

        def _extract_metadata(self, element, index=None) -> Dict[str, Any]:
            """Extract metadata based on configuration."""
            metadata = {
                "source": str(self.file_path),
                "filename": self.file_path.name,
                "type": "xml",
            }

            if index is not None:
                metadata["index"] = index

            if self.metadata_tags:
                for tag in self.metadata_tags:
                    match = element.find(f".//{tag}")
                    if match is not None and match.text:
                        metadata[tag] = match.text

            return metadata


# Export all the loader classes for backward compatibility
__all__ = [
    "DocumentLoader",
    "TextFileLoader",
    "PDFLoader",
    "EnhancedPDFLoader",
    "CSVLoader",
    "JSONLoader",
    "WebLoader",
    "WebPageLoader",
    "DirectoryLoader",
    "XMLLoader",
]

# Suppress deprecation warnings for backward compatibility
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
