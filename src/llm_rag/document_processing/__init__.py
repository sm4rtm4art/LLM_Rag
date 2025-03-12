"""Document processing module for the LLM-RAG system.

This module provides components for loading, processing, and chunking documents
for use in retrieval-augmented generation pipelines.
"""

import warnings
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
    """Abstract base class for document loaders.

    DEPRECATED: Use the DocumentLoader from
    llm_rag.document_processing.loaders instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the DocumentLoader with a deprecation warning.

        This constructor is deliberately kept simple as this class is
        deprecated.
        """
        warnings.warn(
            "This DocumentLoader class is deprecated. "
            "Use the DocumentLoader from "
            "llm_rag.document_processing.loaders instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()

    @abstractmethod
    def load(self) -> Documents:
        """Load documents from a source.

        Returns
        -------
            List of documents, where each document is a dictionary with
            'content' and 'metadata' keys.

        """
        pass


# Import the new modular loaders
from .loaders import (  # noqa: E402
    CSVLoader,
    DirectoryLoader,
    EnhancedPDFLoader,
    JSONLoader,
    PDFLoader,
    TextFileLoader,
    WebLoader,
    WebPageLoader,
    XMLLoader,
)

# Show deprecation warning for directly importing from this module
warnings.warn(
    "Importing loaders directly from llm_rag.document_processing "
    "is deprecated. Please import them from "
    "llm_rag.document_processing.loaders instead.",
    DeprecationWarning,
    stacklevel=2,
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
    "WebLoader",
    "WebPageLoader",
    "XMLLoader",
]
