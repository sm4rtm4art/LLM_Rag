"""Document processing utilities for RAG pipelines.

This module provides functions for processing documents in the RAG pipeline.
It handles standardization of document formats and metadata extraction.
"""

from typing import Any, Dict, List, Optional

from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


def _process_document(document: Any) -> Optional[Dict[str, Any]]:
    """Process a single document from various sources into a standardized format.

    This function extracts content and metadata from documents of various types
    and converts them to a standardized dictionary format.

    Args:
        document: The document to process

    Returns:
        A dictionary with standardized format or None if processing fails

    """
    try:
        # Handle different document types
        if hasattr(document, "page_content") and hasattr(document, "metadata"):
            # Standard LangChain document
            return {
                "content": document.page_content,
                "metadata": document.metadata,
            }
        elif isinstance(document, dict) and "content" in document:
            # Dictionary with content field
            metadata = document.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"raw_metadata": str(metadata)}
            return {
                "content": document["content"],
                "metadata": metadata,
            }
        elif isinstance(document, dict) and "page_content" in document:
            # Dictionary with page_content field (LangChain style)
            metadata = document.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"raw_metadata": str(metadata)}
            return {
                "content": document["page_content"],
                "metadata": metadata,
            }
        elif isinstance(document, dict) and "text" in document:
            # Dictionary with text field
            metadata = document.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"raw_metadata": str(metadata)}
            return {
                "content": document["text"],
                "metadata": metadata,
            }
        elif isinstance(document, str):
            # Plain text
            return {
                "content": document,
                "metadata": {},
            }
        else:
            # Try to extract content and metadata using common attribute patterns
            content = None
            metadata = {}

            # Check for content attributes
            for attr in ["content", "page_content", "text", "body"]:
                if hasattr(document, attr):
                    content = getattr(document, attr)
                    break

            # Check for metadata attributes
            if hasattr(document, "metadata"):
                metadata = document.metadata
                if not isinstance(metadata, dict):
                    metadata = {"raw_metadata": str(metadata)}

            # If content was found, return the processed document
            if content is not None:
                return {
                    "content": content,
                    "metadata": metadata,
                }

            # Last resort: try to convert the entire document to a string
            return {
                "content": str(document),
                "metadata": {},
            }
    except Exception as e:
        logger.warning(f"Error processing document: {str(e)}")
        return None


def _process_documents(documents: Any) -> List[Dict[str, Any]]:
    """Process a list of documents into a standardized format.

    Args:
        documents: List of documents to process

    Returns:
        List of processed documents in standardized format

    """
    if not documents:
        return []

    processed_documents = []

    # Handle different collection types
    if isinstance(documents, dict):
        # Single document as dictionary
        processed = _process_document(documents)
        if processed:
            processed_documents.append(processed)
    else:
        # Collection of documents
        try:
            for doc in documents:
                processed = _process_document(doc)
                if processed:
                    processed_documents.append(processed)
        except TypeError:
            # Not iterable, try processing as a single document
            processed = _process_document(documents)
            if processed:
                processed_documents.append(processed)

    return processed_documents
