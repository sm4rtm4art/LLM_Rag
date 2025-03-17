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
from .file_loaders import CSVLoader, TextFileLoader, XMLLoader
from .web_loader import WebLoader, WebPageLoader

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
    "TextFileLoader",
    "XMLLoader",
    "WebLoader",
    "WebPageLoader",
]
