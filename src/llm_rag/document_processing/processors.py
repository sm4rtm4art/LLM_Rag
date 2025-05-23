"""Document processing utilities for the RAG system."""

from typing import Any, Dict, List, Optional, TypeAlias, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Type aliases to improve readability and avoid long lines
DocumentMetadata: TypeAlias = Dict[str, Any]
DocumentContent: TypeAlias = Union[str, DocumentMetadata]
Document: TypeAlias = Dict[str, DocumentContent]
Documents: TypeAlias = List[Document]


class TextSplitter:
    """Split text into chunks for processing."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        separator: Optional[str] = None,
    ):
        """Initialize the text splitter.

        Args:
        ----
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            separators: List of separators to use for splitting text
            separator: Single separator to use for splitting text (deprecated)
                       If provided, it will be used to create a single-item list for separators

        """
        # Handle both separators and separator parameters for backward compatibility
        if separator is not None and separators is None:
            # Support the old interface with a single separator
            separators = [separator]
            import warnings

            warnings.warn(
                "The 'separator' parameter is deprecated. Use 'separators' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if separators is None:
            separators = ['\n\n', '\n', ' ', '']

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

        # Store separators for testing purposes
        self._separators = separators

    @property
    def separators(self) -> List[str]:
        """Get the separators used by this splitter.

        Returns
        -------
            List of separators

        """
        return self._separators

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
        ----
            text: Text to split

        Returns:
        -------
            List of text chunks

        """
        return self.splitter.split_text(text)

    def split_documents(self, documents: Documents) -> Documents:
        """Split documents into chunks.

        Args:
        ----
            documents: List of documents to split

        Returns:
        -------
            List of document chunks

        """
        # Convert to LangChain document format
        from langchain.schema import Document as LangChainDocument

        lc_docs = [
            LangChainDocument(
                page_content=(str(doc['content']) if not isinstance(doc['content'], str) else doc['content']),
                metadata=doc['metadata'],
            )
            for doc in documents
        ]

        # Split documents
        split_docs = self.splitter.split_documents(lc_docs)

        # Convert back to our format
        return [{'content': doc.page_content, 'metadata': doc.metadata} for doc in split_docs]


class DocumentProcessor:
    """Process documents for the RAG system."""

    def __init__(self, text_splitter: TextSplitter):
        """Initialize the document processor.

        Args:
        ----
            text_splitter: Text splitter to use for chunking documents

        """
        self.text_splitter = text_splitter

    def process(self, documents: Documents) -> Documents:
        """Process documents for the RAG system.

        Args:
        ----
            documents: List of documents to process

        Returns:
        -------
            List of processed documents

        """
        # Filter out documents with empty content
        filtered_docs = [
            doc
            for doc in documents
            if (
                doc.get('content')
                and (
                    (isinstance(doc['content'], str) and doc['content'].strip())
                    or (not isinstance(doc['content'], str))
                )
            )
        ]

        # Split documents into chunks
        chunked_docs = self.text_splitter.split_documents(filtered_docs)

        return chunked_docs
