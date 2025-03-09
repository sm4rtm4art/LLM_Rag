"""Document processing module for the RAG system.

This module contains components for loading and processing documents
for use in RAG pipelines.
"""

import importlib
from typing import Optional

from llm_rag.document_processing.chunking import (
    CharacterTextChunker,
    RecursiveTextChunker,
)

# Import the base loaders that don't have optional dependencies
from llm_rag.document_processing.loaders import (
    CSVLoader,
    DirectoryLoader,
    DocumentLoader,
    PDFLoader,
    TextFileLoader,
)

from llm_rag.document_processing.processors import (
    DocumentProcessor,
    TextSplitter,
)

# Initialize __all__ with the classes that don't have optional dependencies
__all__ = [
    "CSVLoader",
    "DirectoryLoader",
    "DocumentLoader",
    "PDFLoader",
    "TextFileLoader",
    "DocumentProcessor",
    "TextSplitter",
    "CharacterTextChunker",
    "RecursiveTextChunker",
]


# Check for optional dependencies and conditionally import classes
def _import_optional(module_name: str, class_name: str) -> Optional[type]:
    """Import optional class if its requirements are available."""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, class_name):
            __all__.append(class_name)
            return getattr(module, class_name)
    except (ImportError, AttributeError):
        pass
    return None


# Import optional loader classes if available
EnhancedPDFLoader = _import_optional(
    "llm_rag.document_processing.loaders", "EnhancedPDFLoader"
)
JSONLoader = _import_optional(
    "llm_rag.document_processing.loaders", "JSONLoader"
)
WebPageLoader = _import_optional(
    "llm_rag.document_processing.loaders", "WebPageLoader"
)
