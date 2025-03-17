"""Document loaders for various file formats.

This module includes components for loading documents from various sources and formats.
It provides loaders for PDFs, text files, CSV files, JSON, and web content.
"""

from pathlib import Path
from typing import List

from .base import DocumentLoader, FileLoader, LoaderRegistry, registry
from .directory_loader import DirectoryLoader, load_document
from .file_loaders import CSVLoader, TextFileLoader, XMLLoader
from .pdf_loaders import PDFLoader
from .web_loader import WebLoader, WebPageLoader


def load_documents_from_directory(directory_path: str) -> List[dict]:
    """Load all documents from a directory using appropriate loaders.

    Args:
        directory_path: Path to the directory containing documents

    Returns:
        List of loaded documents

    Raises:
        NotADirectoryError: If the directory does not exist or is not a directory

    """
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory_path}")

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
