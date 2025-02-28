"""Document processing utilities for the RAG system."""

from typing import Dict, List, Optional, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    """Split text into chunks for processing."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """Initialize the text splitter.

        Args:
        ----
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            separators: List of separators to use for splitting text

        """
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

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

    def split_documents(self, documents: List[Dict[str, Union[str, Dict]]]) -> List[Dict[str, Union[str, Dict]]]:
        """Split documents into chunks.

        Args:
        ----
            documents: List of documents to split

        Returns:
        -------
            List of document chunks

        """
        # Convert to LangChain document format
        from langchain.schema import Document

        lc_docs = [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in documents]

        # Split documents
        split_docs = self.splitter.split_documents(lc_docs)

        # Convert back to our format
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in split_docs]


class DocumentProcessor:
    """Process documents for the RAG system."""

    def __init__(self, text_splitter: TextSplitter):
        """Initialize the document processor.

        Args:
        ----
            text_splitter: Text splitter to use for chunking documents

        """
        self.text_splitter = text_splitter

    def process(self, documents: List[Dict[str, Union[str, Dict]]]) -> List[Dict[str, Union[str, Dict]]]:
        """Process documents for the RAG system.

        Args:
        ----
            documents: List of documents to process

        Returns:
        -------
            List of processed documents

        """
        # Filter out documents with empty content
        filtered_docs = [doc for doc in documents if doc.get("content") and doc["content"].strip()]

        # Split documents into chunks
        chunked_docs = self.text_splitter.split_documents(filtered_docs)

        return chunked_docs
