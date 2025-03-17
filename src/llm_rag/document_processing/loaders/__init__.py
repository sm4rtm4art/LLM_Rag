"""Document loaders for various file formats.

This module includes components for loading documents from various sources and formats.
It provides loaders for PDFs, text files, CSV files, JSON, and web content.
"""

from .base import DirectoryLoader, DocumentLoader, FileLoader
from .file_loaders import CSVLoader, TextFileLoader, XMLLoader
from .pdf_loaders import PDFLoader
from .web_loader import WebLoader, WebPageLoader


def load_document(file_path: str) -> str:
    """Load a document using the appropriate loader based on the file path."""
    # Implementation will be added later
    pass


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
    "load_document",
]
