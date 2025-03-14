"""Document processing module for the LLM-RAG system.

This module provides components for loading, processing, and chunking documents
for use in retrieval-augmented generation pipelines.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeAlias, Union

# Re-export from existing modules for backward compatibility
from .chunking import CharacterTextChunker, RecursiveTextChunker  # noqa: F401
from .processors import DocumentProcessor, TextSplitter  # noqa: F401

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


# Import the new modular loaders with error handling in case they're not available
try:
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

    _has_loaders = True
except ImportError as e:
    warnings.warn(
        f"Error importing document loaders: {str(e)}. Some functionality may be limited.",
        ImportWarning,
        stacklevel=2,
    )
    _has_loaders = False

# Show deprecation warning for directly importing from this module
warnings.warn(
    "Importing loaders directly from llm_rag.document_processing "
    "is deprecated. Please import them from "
    "llm_rag.document_processing.loaders instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Define the exports based on what's available
_common_exports = [
    "DocumentMetadata",
    "DocumentContent",
    "Document",
    "Documents",
    "DocumentLoader",
    "DocumentProcessor",
    "TextSplitter",
    "CharacterTextChunker",
    "RecursiveTextChunker",
]

if _has_loaders:
    __all__ = _common_exports + [
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
else:
    __all__ = _common_exports
