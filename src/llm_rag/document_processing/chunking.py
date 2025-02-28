"""Text chunking module for the llm-rag package.

This module provides utilities for splitting documents into manageable chunks.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class TextChunker(ABC):
    """Abstract base class for text chunkers.

    Text chunkers are responsible for splitting documents into smaller chunks
    that can be processed by the embedding model and stored in the vector store.
    """

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
        ----
            text: The text to split.

        Returns:
        -------
            A list of text chunks.

        """
        pass

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks.

        Args:
        ----
            documents: List of documents to split, where each document is
                a dictionary with 'content' and 'metadata' keys.

        Returns:
        -------
            A list of document chunks, where each chunk is a dictionary with
                'content' and 'metadata' keys.

        """
        chunked_documents = []
        for doc in documents:
            content = doc.get("content", "")
            metadata: Dict[str, Any] = doc.get("metadata", {})

            if not content:
                continue

            chunks = self.split_text(content)

            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["chunk_count"] = len(chunks)

                chunked_documents.append(
                    {
                        "content": chunk,
                        "metadata": chunk_metadata,
                    }
                )

        return chunked_documents


class CharacterTextChunker(TextChunker):
    """Chunker that splits text by character count."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = " ",
    ) -> None:
        """Initialize the character text chunker.

        Args:
        ----
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters to overlap between chunks.
            separator: String to use as separator when joining chunks.

        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
        ----
            text: The text to split.

        Returns:
        -------
            A list of text chunks.

        """
        # If text is shorter than chunk_size, return it as is
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))

            if end >= len(text):
                # If we've reached the end, just add the remaining text
                chunks.append(text[start:])
                break

            # Find a good split point (at a separator)
            split_point = text.rfind(self.separator, start, end)

            if split_point == -1 or split_point <= start:
                # If no separator found, just split at chunk_size
                split_point = end

            # Add the chunk
            chunks.append(text[start:split_point])

            # Move start position for next chunk, accounting for overlap
            # Ensure we always make progress by moving at least one character forward
            start = max(start + 1, split_point - self.chunk_overlap)

        return chunks


class RecursiveTextChunker(TextChunker):
    """Chunker that splits text based on separators."""

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize the recursive text chunker.

        Args:
        ----
            separators: List of separators to use for splitting text,
                in order of priority. If None, defaults to common separators.
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters to overlap between chunks.

        """
        super().__init__()
        if separators is None:
            separators = ["\n\n", "\n", ". ", ", ", " ", ""]

        # Add validation for chunk_size and chunk_overlap
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.separators = separators
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
        ----
            text: The text to split.

        Returns:
        -------
            A list of text chunks.

        """
        # If text is shorter than chunk_size, return it as is
        if len(text) <= self.chunk_size:
            return [text]

        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                # Split by this separator
                splits = text.split(separator)

                # Filter out empty splits and add separator back
                splits_with_separator = []
                for i, split in enumerate(splits):
                    if not split:
                        continue
                    # Add separator back except for the last item
                    if i < len(splits) - 1 or text.endswith(separator):
                        splits_with_separator.append(split + separator)
                    else:
                        splits_with_separator.append(split)

                # If splitting didn't help, try the next separator
                if len(splits_with_separator) <= 1:
                    continue

                # Merge splits into chunks that respect chunk_size
                chunks = []
                current_chunk: List[str] = []
                current_length = 0

                for split in splits_with_separator:
                    # If adding this split would exceed chunk_size, start a new chunk
                    if current_length + len(split) > self.chunk_size:
                        if current_chunk:
                            chunks.append("".join(current_chunk))

                        # If a single split is too large, recursively split it
                        if len(split) > self.chunk_size:
                            # Use character chunking as a fallback
                            char_chunker = CharacterTextChunker(
                                chunk_size=self.chunk_size,
                                chunk_overlap=self.chunk_overlap,
                            )
                            chunks.extend(char_chunker.split_text(split))
                        else:
                            current_chunk = [split]
                            current_length = len(split)
                    else:
                        current_chunk.append(split)
                        current_length += len(split)

                # Add the last chunk if there's anything left
                if current_chunk:
                    chunks.append("".join(current_chunk))

                # If we successfully created chunks, add overlaps
                if chunks:
                    # Add overlaps if needed
                    if self.chunk_overlap > 0 and len(chunks) > 1:
                        overlapped_chunks = []
                        for i, chunk in enumerate(chunks):
                            if i == 0:
                                overlapped_chunks.append(chunk)
                            else:
                                # Extract overlap from previous chunk
                                prev_chunk = overlapped_chunks[i - 1]
                                overlap_chars = min(self.chunk_overlap, len(prev_chunk))
                                overlap_text = prev_chunk[-overlap_chars:]

                                # Add overlap text to beginning of current chunk
                                overlapped_chunks.append(overlap_text + chunk)

                        chunks = overlapped_chunks

                    return chunks

        # If no separator worked, fall back to character chunking
        char_chunker = CharacterTextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return char_chunker.split_text(text)
