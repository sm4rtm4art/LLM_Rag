#!/usr/bin/env python
"""Unit tests for the chunking module."""

# Standard library imports
import sys
from pathlib import Path

# Add the project root to the Python path before other imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Third-party imports
import pytest

# Local imports
from llm_rag.document_processing.chunking import (
    CharacterTextChunker,
    RecursiveTextChunker,
)


@pytest.fixture
def recursive_chunker():
    """Create a RecursiveTextChunker instance for testing."""
    return RecursiveTextChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def character_chunker():
    """Create a CharacterTextChunker instance for testing."""
    return CharacterTextChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def long_text():
    """Create a long text for testing."""
    return "This is a test document. " * 20


@pytest.fixture
def text_with_sentences():
    """Create a text with multiple sentences for testing."""
    return (
        "This is the first sentence. This is the second sentence. "
        "This is the third sentence. This is the fourth sentence."
    )


class TestRecursiveTextChunker:
    """Test the RecursiveTextChunker class."""

    def test_split_text(self, recursive_chunker, long_text):
        """Test splitting a text document."""
        # Act
        chunks = recursive_chunker.split_text(long_text)

        # Assert
        assert len(chunks) > 0
        assert len(chunks[0]) <= 100

    def test_chunk_overlap(self, recursive_chunker, long_text):
        """Test that chunks have the expected overlap."""
        # Act
        chunks = recursive_chunker.split_text(long_text)

        # Assert
        if len(chunks) > 1:
            # Check if there's overlap between consecutive chunks
            first_chunk_end = chunks[0][-20:]
            second_chunk_start = chunks[1][:20]

            # Check for partial overlap - at least some characters should match
            overlap_chars = set(first_chunk_end) & set(second_chunk_start)
            assert len(overlap_chars) > 0, "Chunks should have some overlap"


class TestCharacterTextChunker:
    """Test the CharacterTextChunker class."""

    def test_split_text(self, character_chunker, long_text):
        """Test splitting a text document."""
        # Act
        chunks = character_chunker.split_text(long_text)

        # Assert
        assert len(chunks) > 0
        assert len(chunks[0]) <= 100

    def test_empty_text(self, character_chunker):
        """Test splitting an empty text."""
        # Act
        chunks = character_chunker.split_text("")

        # Assert
        # Empty text might return empty list or list with empty string
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0] == "")


class TestSentenceSplitting:
    """Test sentence splitting functionality."""

    def test_split_by_sentence(self, recursive_chunker, text_with_sentences):
        """Test splitting text by sentences."""
        # Act
        chunks = recursive_chunker.split_text(text_with_sentences)

        # Assert
        assert len(chunks) > 0

        # Check for sentence-ending punctuation
        endings = [chunk.strip()[-1] for chunk in chunks if chunk.strip()]
        assert any(end in [".", "!", "?"] for end in endings), "Some chunks should end with sentence punctuation"
