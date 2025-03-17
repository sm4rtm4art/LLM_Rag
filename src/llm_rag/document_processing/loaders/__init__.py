"""Document loaders for various file formats.

This module includes components for loading documents from various sources and formats.
It provides loaders for PDFs, text files, CSV files, JSON, and web content.
"""

from pathlib import Path
from typing import List

from .base import DocumentLoader, FileLoader, LoaderRegistry, registry
from .directory_loader import DirectoryLoader
from .file_loaders import CSVLoader, TextFileLoader, XMLLoader
from .pdf_loaders import PDFLoader
from .web_loader import WebLoader, WebPageLoader


def load_document(file_path: str) -> List[dict]:
    """Load a document using the appropriate loader based on the file path.

    Args:
        file_path: Path to the document to load

    Returns:
        List of documents loaded from the file

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If no loader is found for the file type

    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = registry.create_loader_for_file(path)
    if loader is None:
        raise ValueError(f"No loader found for file type: {path.suffix}")

    return loader.load_from_file(path)


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
