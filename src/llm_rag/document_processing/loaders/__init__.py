"""Document loaders implementation modules.

This package contains the implementation of document loaders
for different document types.
"""

# Import components from submodules
from .base import DocumentLoader, FileLoader, LoaderRegistry, registry
from .directory_loader import DirectoryLoader, load_documents_from_directory
from .factory import get_available_loader_extensions, load_document
from .file_loaders import CSVLoader, EnhancedPDFLoader, JSONLoader, PDFLoader, TextFileLoader, XMLLoader
from .web_loaders import WebLoader, WebPageLoader

# Flag for indicating module import success (used by stub implementation)
_MODULAR_IMPORT_SUCCESS = True

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
    "load_documents_from_directory",
    "load_document",
    "get_available_loader_extensions",
    "_MODULAR_IMPORT_SUCCESS",
]
