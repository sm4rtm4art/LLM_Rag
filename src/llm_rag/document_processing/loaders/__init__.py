"""Document loaders for the LLM-RAG system.

This module provides components for loading documents from various sources
and formats, including PDFs, text files, CSV files, JSON, and web content.
"""

# Import base classes and interfaces
from .base import DirectoryLoader as BaseDirectoryLoader
from .base import DocumentLoader, FileLoader, LoaderRegistry, registry
from .base import WebLoader as BaseWebLoader

# Import concrete loader implementations
from .directory_loader import DirectoryLoader
from .file_loaders import CSVLoader, PDFLoader, TextFileLoader, XMLLoader
from .web_loader import WebLoader, WebPageLoader


def load_document(file_path: str, **kwargs) -> DocumentLoader:
    """Load a document using the appropriate loader.

    Args:
        file_path: Path to the document to load
        **kwargs: Additional arguments to pass to the loader

    Returns:
        A DocumentLoader instance for the document

    """
    return registry.get_loader(file_path)(file_path, **kwargs)


# Re-export all components
__all__ = [
    # Base classes
    "DocumentLoader",
    "FileLoader",
    "BaseDirectoryLoader",
    "BaseWebLoader",
    "LoaderRegistry",
    "registry",
    # Concrete implementations
    "DirectoryLoader",
    "CSVLoader",
    "PDFLoader",
    "TextFileLoader",
    "XMLLoader",
    "WebLoader",
    "WebPageLoader",
    # Utility functions
    "load_document",
]
