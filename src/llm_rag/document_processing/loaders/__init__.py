"""Document loaders for various file formats.

This module includes components for loading documents from various sources and formats.
It provides loaders for PDFs, text files, CSV files, JSON, and web content.
"""

from .base import DirectoryLoader, DocumentLoader, FileLoader, LoaderRegistry, registry
from .file_loaders import CSVLoader, TextFileLoader, XMLLoader
from .pdf_loaders import PDFLoader
from .web_loader import WebLoader, WebPageLoader


def load_document(file_path: str) -> str:
    """Load a document using the appropriate loader based on the file path."""
    # Implementation will be added later
    pass


def load_documents_from_directory(directory_path: str) -> list[str]:
    """Load all documents from a directory using appropriate loaders.

    Args:
        directory_path: Path to the directory containing documents

    Returns:
        List of loaded document contents

    """
    loader = DirectoryLoader(directory_path)
    return loader.load()


__all__ = [
    "DocumentLoader",
    "FileLoader",
    "DirectoryLoader",
    "CSVLoader",
    "PDFLoader",
    "TextFileLoader",
    "XMLLoader",
    "WebLoader",
    "WebPageLoader",
    "LoaderRegistry",
    "registry",
    "load_document",
    "load_documents_from_directory",
]
