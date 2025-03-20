"""Document loaders for the LLM-RAG system.

This module provides components for loading documents from various sources
and formats, including PDFs, text files, CSV files, JSON, and web content.
"""

from pathlib import Path
from typing import List

from .base import DocumentLoader, FileLoader, LoaderRegistry, registry
from .directory_loader import DirectoryLoader
from .file_loaders import CSVLoader, EnhancedPDFLoader, JSONLoader, PDFLoader, TextFileLoader, XMLLoader
from .web_loaders import WebLoader, WebPageLoader


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
    "EnhancedPDFLoader",
    "JSONLoader",
    "PDFLoader",
    "TextFileLoader",
    "WebLoader",
    "WebPageLoader",
    "XMLLoader",
    "LoaderRegistry",
    "registry",
    "load_documents_from_directory",
]
