"""Document loaders implementation modules.

This package contains the implementation of document loaders
for different document types.
"""

# Import components from submodules
from .base import DocumentLoader, FileLoader, LoaderRegistry, registry
from .directory_loader import DirectoryLoader
from .file_loaders import CSVLoader, EnhancedPDFLoader, JSONLoader, PDFLoader, TextFileLoader, XMLLoader
from .web_loaders import WebLoader, WebPageLoader

# Re-export all components
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
]
