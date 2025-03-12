"""Document processing module for the LLM-RAG system.

This module provides components for loading, processing, and chunking documents
for use in retrieval-augmented generation pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeAlias, Union

# Re-export from existing modules for backward compatibility
from .chunking import CharacterTextChunker, RecursiveTextChunker
from .processors import DocumentProcessor, TextSplitter

# Type aliases to improve readability and avoid long lines
DocumentMetadata: TypeAlias = Dict[str, Any]
DocumentContent: TypeAlias = Union[str, DocumentMetadata]
Document: TypeAlias = Dict[str, DocumentContent]
Documents: TypeAlias = List[Document]


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self) -> Documents:
        """Load documents from a source.

        Returns
        -------
            List of documents, where each document is a dictionary with
            'content' and 'metadata' keys.

        """
        pass


# Import and re-export all from legacy loaders to maintain backward compatibility
from .loaders import (  # noqa: E402
    CSVLoader,
    DirectoryLoader,
    EnhancedPDFLoader,
    JSONLoader,
    PDFLoader,
    TextFileLoader,
    WebPageLoader,
)

# These will be added in the new modular structure
__all__ = [
    "DocumentMetadata",
    "DocumentContent",
    "Document",
    "Documents",
    "DocumentLoader",
    "DocumentProcessor",
    "TextSplitter",
    "CharacterTextChunker",
    "RecursiveTextChunker",
    "CSVLoader",
    "DirectoryLoader",
    "EnhancedPDFLoader",
    "JSONLoader",
    "PDFLoader",
    "TextFileLoader",
    "WebPageLoader",
]
