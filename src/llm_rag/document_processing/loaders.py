"""Document loaders for the LLM-RAG system.

This module provides components for loading documents from various sources
and formats, including PDFs, text files, CSV files, JSON, and web content.

Note: This file is maintained for backward compatibility. For new development,
please use the modular components in the loaders/ directory.
"""

# Standard library imports
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    # First try to import from the modular implementation
    from llm_rag.document_processing.loaders.base import DocumentLoader, FileLoader, LoaderRegistry, registry
    from llm_rag.document_processing.loaders.directory_loader import DirectoryLoader, load_documents_from_directory
    from llm_rag.document_processing.loaders.factory import get_available_loader_extensions, load_document
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
    # If import fails, use stub classes
    warnings.warn(
        f"Failed to import modular document loaders: {e}. Using stub classes.",
        stacklevel=2,
    )
    _MODULAR_IMPORT_SUCCESS = False

# Local imports
from llm_rag.utils.logging import get_logger

# Get configured logger
logger = get_logger(__name__)

# Define document types
Documents = List[Dict[str, Any]]


# Define adapter classes for backward compatibility
if _MODULAR_IMPORT_SUCCESS:
    # Import was successful, just re-export the components

    # Define the load_document function for backward compatibility if different
    def load_document(file_path: Union[str, Path], exclude_patterns: Optional[List[str]] = None) -> Optional[Documents]:
        """Load a document from a file.

        This is a compatibility wrapper around the modular implementation.

        Args:
            file_path: Path to the file to load
            exclude_patterns: Optional list of glob patterns to exclude

        Returns:
            List of documents or None if the file couldn't be loaded

        """
        try:
            from llm_rag.document_processing.loaders.factory import load_document as _load_document

            return _load_document(file_path, exclude_patterns)
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return None

else:
    # Import failed, define stub classes

    class DocumentLoader:
        """Stub class for DocumentLoader."""

        def load(self) -> Documents:
            """Stub method."""
            return []

    class FileLoader(DocumentLoader):
        """Stub class for FileLoader."""

        def __init__(self, file_path: Optional[Union[str, Path]] = None):
            """Initialize with a file path."""
            self.file_path = file_path

        def load_from_file(self, file_path: Union[str, Path]) -> Documents:
            """Stub method."""
            return []

    class LoaderRegistry:
        """Stub class for LoaderRegistry."""

        def register_loader(self, extension, loader_class):
            """Stub method."""
            pass

        def create_loader_for_file(self, file_path):
            """Stub method."""
            return None

    # Create a stub registry instance
    registry = LoaderRegistry()

    class DirectoryLoader(DocumentLoader):
        """Stub class for DirectoryLoader."""

        def __init__(self, directory_path: Union[str, Path]):
            """Initialize with a directory path."""
            self.directory_path = directory_path

        def load(self) -> Documents:
            """Stub method."""
            return []

    class CSVLoader(FileLoader):
        """Stub class for CSVLoader."""

        pass

    class PDFLoader(FileLoader):
        """Stub class for PDFLoader."""

        pass

    class EnhancedPDFLoader(PDFLoader):
        """Stub class for EnhancedPDFLoader."""

        pass

    class JSONLoader(FileLoader):
        """Stub class for JSONLoader."""

        pass

    class TextFileLoader(FileLoader):
        """Stub class for TextFileLoader."""

        pass

    class XMLLoader(FileLoader):
        """Stub class for XMLLoader."""

        pass

    class WebLoader(DocumentLoader):
        """Stub class for WebLoader."""

        pass

    class WebPageLoader(WebLoader):
        """Stub class for WebPageLoader."""

        pass

    def load_documents_from_directory(
        directory_path: Union[str, Path], exclude_patterns: Optional[List[str]] = None
    ) -> Documents:
        """Stub function for loading documents from a directory."""
        return []

    def load_document(file_path: Union[str, Path], exclude_patterns: Optional[List[str]] = None) -> Optional[Documents]:
        """Stub function for loading a document."""
        return None

    def get_available_loader_extensions() -> Dict[str, str]:
        """Stub function for getting available loader extensions."""
        return {}


# Define exports
__all__ = [
    "DocumentLoader",
    "FileLoader",
    "LoaderRegistry",
    "registry",
    "DirectoryLoader",
    "CSVLoader",
    "EnhancedPDFLoader",
    "JSONLoader",
    "PDFLoader",
    "TextFileLoader",
    "XMLLoader",
    "WebLoader",
    "WebPageLoader",
    "load_documents_from_directory",
    "load_document",
    "get_available_loader_extensions",
    "Documents",
]

# Suppress deprecation warnings for backward compatibility
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
