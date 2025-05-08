"""Tests for document chunking functionality."""

import pytest

from llm_rag.document_processing.chunking import (
    CharacterTextChunker,
    RecursiveTextChunker,
)


def test_character_chunker_basic():
    """Test basic functionality of character chunker."""
    # Given
    chunker = CharacterTextChunker(chunk_size=10, chunk_overlap=2)
    text = 'This is a test text for chunking'

    # When
    chunks = chunker.split_text(text)

    # Then
    assert len(chunks) > 1
    # First chunk should be at most chunk_size characters
    assert len(chunks[0]) <= 10

    # Check content is preserved (accounting for possible different chunk boundaries)
    joined = ''.join(chunks)
    # The joined chunks may have duplicated content at the overlaps
    assert len(joined) >= len(text)


def test_recursive_chunker_basic():
    """Test basic functionality of recursive chunker."""
    # Given
    # Increase chunk_size to 30 to accommodate the longest sentence (26 chars)
    chunker = RecursiveTextChunker(chunk_size=30, chunk_overlap=5)
    text = 'This is a test. It has multiple sentences. Some are short. Others might be longer.'

    # When
    chunks = chunker.split_text(text)

    # Then
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= 35  # chunk_size + possible overlap

    # Verify we can recover original sentences
    for sentence in ['This is a test.', 'It has multiple sentences.', 'Some are short.', 'Others might be longer.']:
        found = False
        for chunk in chunks:
            if sentence in chunk:
                found = True
                break
        assert found, f'Sentence not found in any chunk: {sentence}'


def test_split_documents():
    """Test splitting documents with metadata."""
    # Given
    chunker = CharacterTextChunker(chunk_size=10, chunk_overlap=0)
    documents = [
        {
            'content': 'This is a longer document that should be split',
            'metadata': {'source': 'test.txt', 'custom': 'value'},
        }
    ]

    # When
    chunked_docs = chunker.split_documents(documents)

    # Then
    assert len(chunked_docs) > 1

    # Check metadata is preserved and augmented
    for i, doc in enumerate(chunked_docs):
        assert 'metadata' in doc
        assert doc['metadata']['source'] == 'test.txt'
        assert doc['metadata']['custom'] == 'value'
        assert doc['metadata']['chunk_index'] == i
        assert doc['metadata']['chunk_count'] == len(chunked_docs)


def test_chunker_validation():
    """Test chunker validation logic."""
    # Then - Should raise errors for invalid parameters
    with pytest.raises(ValueError):
        CharacterTextChunker(chunk_size=0)

    with pytest.raises(ValueError):
        CharacterTextChunker(chunk_overlap=-1)

    with pytest.raises(ValueError):
        CharacterTextChunker(chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError):
        RecursiveTextChunker(chunk_size=10, chunk_overlap=20)
