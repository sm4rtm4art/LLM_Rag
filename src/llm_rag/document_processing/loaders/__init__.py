"""Document loaders for the LLM-RAG system.

This module provides document loaders for different file types and sources.
All loaders implement the DocumentLoader interface and return standardized
documents.
"""

# Import the legacy loaders directly from the loaders.py file
# to avoid circular imports
import importlib.util
import sys
from pathlib import Path

# Get the path to the legacy loaders.py file
loaders_path = Path(__file__).parent.parent / "loaders.py"
spec = importlib.util.spec_from_file_location("legacy_loaders", loaders_path)
legacy_loaders = importlib.util.module_from_spec(spec)
sys.modules["legacy_loaders"] = legacy_loaders
spec.loader.exec_module(legacy_loaders)

# Create aliases for backward compatibility
LegacyCSVLoader = legacy_loaders.CSVLoader
LegacyDirectoryLoader = legacy_loaders.DirectoryLoader
LegacyDocumentLoader = legacy_loaders.DocumentLoader
LegacyEnhancedPDFLoader = legacy_loaders.EnhancedPDFLoader
LegacyJSONLoader = legacy_loaders.JSONLoader
LegacyPDFLoader = legacy_loaders.PDFLoader
LegacyTextFileLoader = legacy_loaders.TextFileLoader
LegacyWebPageLoader = legacy_loaders.WebPageLoader

# Re-export the base classes and interfaces
from .base import (
    DirectoryLoader as BaseDirectoryLoader,
)
from .base import (
    DocumentLoader,
    FileLoader,
    LoaderRegistry,
    registry,
)
from .base import (
    WebLoader as BaseWebLoader,
)

# Re-export directory loader
from .directory_loader import DirectoryLoader

# Re-export the factory functions
from .factory import (
    get_available_loader_extensions,
    load_document,
    load_documents_from_directory,
)

# Re-export the file loaders
from .file_loaders import (
    CSVLoader,
    TextFileLoader,
)

# Re-export JSON loader
from .json_loader import JSONLoader

# Re-export PDF loaders
from .pdf_loaders import (
    EnhancedPDFLoader,
    PDFLoader,
)

# Re-export web loader
from .web_loader import WebLoader, WebPageLoader

# For backward compatibility and public API access
__all__ = [
    # New modular base classes and interfaces
    "DocumentLoader",
    "FileLoader",
    "BaseDirectoryLoader",
    "BaseWebLoader",
    "LoaderRegistry",
    "registry",
    # New modular loader implementations
    "CSVLoader",
    "DirectoryLoader",
    "JSONLoader",
    "PDFLoader",
    "EnhancedPDFLoader",
    "TextFileLoader",
    "WebLoader",
    "WebPageLoader",
    # Factory functions
    "load_document",
    "load_documents_from_directory",
    "get_available_loader_extensions",
    # Legacy loaders (re-exported for backward compatibility)
    "LegacyCSVLoader",
    "LegacyDirectoryLoader",
    "LegacyDocumentLoader",
    "LegacyEnhancedPDFLoader",
    "LegacyJSONLoader",
    "LegacyPDFLoader",
    "LegacyTextFileLoader",
    "LegacyWebPageLoader",
]
