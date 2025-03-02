"""Document chunking functionality for splitting text into manageable pieces."""

from typing import Any, Dict, List, Optional


class BaseTextChunker:
    """Base class for text chunkers."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Initialize the chunker.

        Args:
        ----
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks

        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
        ----
            text: Text to split

        Returns:
        -------
            List of text chunks

        """
        raise NotImplementedError("Subclasses must implement split_text")

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks, preserving metadata.

        Args:
        ----
            documents: List of documents with 'content' and 'metadata' keys

        Returns:
        -------
            List of chunked documents with updated metadata

        """
        chunked_documents = []
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            chunks = self.split_text(content)

            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["chunk_count"] = len(chunks)

                chunked_documents.append({"content": chunk, "metadata": chunk_metadata})

        return chunked_documents


class CharacterTextChunker(BaseTextChunker):
    """Chunker that splits text by character count."""

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on character count.

        Args:
        ----
            text: Text to split

        Returns:
        -------
            List of text chunks

        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks


class RecursiveTextChunker(BaseTextChunker):
    """Chunker that splits text recursively by sentences then characters."""

    def __init__(self, chunk_size: int, chunk_overlap: int, separators: Optional[List[str]] = None):
        """Initialize the recursive chunker.

        Args:
        ----
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting, in order of preference

        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", ", ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text recursively using separators.

        Args:
        ----
            text: Text to split

        Returns:
        -------
            List of text chunks

        """
        if not text:
            return []

        # If text fits in a single chunk, return it
        if len(text) <= self.chunk_size:
            return [text]

        # Try each separator
        for separator in self.separators:
            if separator == "":
                # If we're at the character level, use character chunking
                chunker = CharacterTextChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
                return chunker.split_text(text)

            # Split by the current separator
            splits = text.split(separator)

            # If splitting produced only one chunk, continue to next separator
            if len(splits) == 1:
                continue

            # Process each split
            chunks: List[str] = []
            current_chunk: List[str] = []
            current_length = 0

            for split in splits:
                split_with_sep = split + separator if split != splits[-1] else split
                split_len = len(split_with_sep)

                # If adding this split exceeds chunk size, finalize current chunk
                if current_length + split_len > self.chunk_size and current_chunk:
                    chunks.append(separator.join(current_chunk))

                    # Handle overlap by keeping some splits for the next chunk
                    overlap_splits: List[str] = []
                    overlap_length = 0

                    # Work backwards through current_chunk to find splits for overlap
                    for s in reversed(current_chunk):
                        s_with_sep = s + separator
                        overlap_len = len(s_with_sep)
                        if overlap_length + overlap_len <= self.chunk_overlap:
                            overlap_splits.insert(0, s)
                            overlap_length += overlap_len
                        else:
                            break

                    current_chunk = overlap_splits
                    current_length = overlap_length

                # Add the current split
                current_chunk.append(split)
                current_length += split_len

                # If we've reached chunk size, finalize the chunk
                if current_length >= self.chunk_size:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0

            # Add any remaining content as the final chunk
            if current_chunk:
                chunks.append(separator.join(current_chunk))

            # If we successfully created chunks with this separator, return them
            if chunks:
                return chunks

        # Fallback to character chunking if no separator worked
        chunker = CharacterTextChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return chunker.split_text(text)
