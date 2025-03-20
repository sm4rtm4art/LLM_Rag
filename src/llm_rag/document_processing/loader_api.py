"""Document loaders for the LLM-RAG system.

This module provides components for loading documents from various sources
and formats, including PDFs, text files, CSV files, JSON, and web content.

This module is a re-export layer providing the document loaders from the
`loaders/` package, maintaining backward compatibility while avoiding
namespace collisions.
"""

# Standard library imports
import warnings
from typing import Any, Dict, List

# Internal imports (using relative imports to avoid namespace collisions)
from llm_rag.utils.logging import get_logger

# Import components from the loaders package
# Use relative import paths to avoid the module/package name collision
from .loaders.base import DocumentLoader, FileLoader, LoaderRegistry, registry
from .loaders.directory_loader import DirectoryLoader, load_documents_from_directory
from .loaders.file_loaders import CSVLoader, EnhancedPDFLoader, JSONLoader, PDFLoader, TextFileLoader, XMLLoader
from .loaders.web_loaders import WebLoader, WebPageLoader

# Setup logger
logger = get_logger(__name__)

# Define document types
Documents = List[Dict[str, Any]]

# Export all the loader classes
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

# Suppress deprecation warnings for backward compatibility
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
