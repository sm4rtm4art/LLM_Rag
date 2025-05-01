"""Tests for the EmbeddingComparisonEngine class."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from llm_rag.document_processing.comparison.alignment import AlignmentPair
from llm_rag.document_processing.comparison.comparison_engine import (
    ComparisonConfig,
    ComparisonResult,
    EmbeddingComparisonEngine,
)
from llm_rag.document_processing.comparison.document_parser import Section, SectionType
from llm_rag.utils.errors import DocumentProcessingError


@pytest.fixture
def mock_sections():
    """Create mock sections for testing."""
    return [
        Section(id="s1", type=SectionType.HEADING, content="Introduction", level=1),
        Section(id="s2", type=SectionType.PARAGRAPH, content="This is a test paragraph with some content.", level=0),
        Section(
            id="s3",
            type=SectionType.PARAGRAPH,
            content="This paragraph has been modified with different text.",
            level=0,
        ),
        Section(id="s4", type=SectionType.HEADING, content="Conclusion", level=1),
        Section(id="s5", type=SectionType.PARAGRAPH, content="Final remarks go here.", level=0),
    ]


@pytest.fixture
def mock_alignment_pairs(mock_sections):
    """Create mock alignment pairs for testing."""
    return [
        # Identical sections - should be SIMILAR
        AlignmentPair(
            source_section=mock_sections[0],
            target_section=mock_sections[0],
            similarity_score=0.0,  # Will be calculated by the engine
        ),
        # Modified paragraph - should be MINOR_CHANGES or MAJOR_CHANGES
        AlignmentPair(
            source_section=mock_sections[1],
            target_section=mock_sections[2],
            similarity_score=0.0,  # Will be calculated by the engine
        ),
        # Source-only section - should be DELETED
        AlignmentPair(
            source_section=mock_sections[3],
            target_section=None,
            is_source_only=True,
        ),
        # Target-only section - should be NEW
        AlignmentPair(
            source_section=None,
            target_section=mock_sections[4],
            is_target_only=True,
        ),
    ]


def test_init_with_default_config():
    """Test initialization with default configuration."""
    engine = EmbeddingComparisonEngine()
    assert isinstance(engine.config, ComparisonConfig)
    assert engine.config.similar_threshold == 0.9
    assert engine._embedding_model is None
    assert isinstance(engine._embedding_cache, dict)


def test_init_with_custom_config():
    """Test initialization with custom configuration."""
    config = ComparisonConfig(
        similar_threshold=0.95,
        minor_change_threshold=0.85,
        embedding_model="custom-model",
    )
    engine = EmbeddingComparisonEngine(config)
    assert engine.config.similar_threshold == 0.95
    assert engine.config.minor_change_threshold == 0.85
    assert engine.config.embedding_model == "custom-model"


def test_compare_sections(mock_alignment_pairs):
    """Test comparing aligned section pairs."""
    engine = EmbeddingComparisonEngine()

    # Mock _calculate_similarity to return predictable values
    engine._calculate_similarity = MagicMock()
    engine._calculate_similarity.return_value = 0.95  # High similarity for first pair
    engine._calculate_similarity.side_effect = [0.95, 0.75]  # Different values for each call

    comparisons = engine.compare_sections(mock_alignment_pairs)

    assert len(comparisons) == 4
    assert comparisons[0].result == ComparisonResult.SIMILAR
    assert comparisons[1].result == ComparisonResult.MINOR_CHANGES
    assert comparisons[2].result == ComparisonResult.DELETED
    assert comparisons[3].result == ComparisonResult.NEW

    # Verify _calculate_similarity was called twice (for pairs with both source and target)
    assert engine._calculate_similarity.call_count == 2


def test_calculate_similarity():
    """Test calculating similarity between two sections."""
    engine = EmbeddingComparisonEngine()

    # Mock _get_embedding to return controlled vectors
    engine._get_embedding = MagicMock()
    # First call returns [1, 0, 0], second call returns [0.707, 0.707, 0] (45 degrees apart)
    engine._get_embedding.side_effect = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.707, 0.707, 0.0]),
    ]

    source = Section(id="s1", type=SectionType.PARAGRAPH, content="Source text")
    target = Section(id="s2", type=SectionType.PARAGRAPH, content="Target text")

    similarity = engine._calculate_similarity(source, target)

    # Expected cosine similarity between vectors at 45 degrees is ~0.7071
    assert 0.7 <= similarity <= 0.72
    assert engine._get_embedding.call_count == 2


def test_get_embedding_cache():
    """Test embedding caching mechanism."""
    engine = EmbeddingComparisonEngine()
    engine._compute_embedding = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))

    # First call should compute embedding
    result1 = engine._get_embedding("test text")
    assert engine._compute_embedding.call_count == 1

    # Second call with same text should use cache
    result2 = engine._get_embedding("test text")
    assert engine._compute_embedding.call_count == 1  # Still 1

    # Vectors should be identical
    assert np.array_equal(result1, result2)

    # Different text should compute new embedding
    engine._get_embedding("different text")
    assert engine._compute_embedding.call_count == 2


def test_compute_embedding():
    """Test computing embeddings for text."""
    engine = EmbeddingComparisonEngine()

    # Try with a simple text
    embedding1 = engine._compute_embedding("Simple test text")
    assert isinstance(embedding1, np.ndarray)
    assert embedding1.shape[0] == 10  # Our mock implementation uses 10-dim vectors

    # Verify same text produces same embedding (deterministic)
    embedding2 = engine._compute_embedding("Simple test text")
    assert np.array_equal(embedding1, embedding2)

    # Different text should produce different embedding
    embedding3 = engine._compute_embedding("Different text")
    assert not np.array_equal(embedding1, embedding3)


def test_classify_similarity():
    """Test classification of similarity scores."""
    engine = EmbeddingComparisonEngine()

    assert engine._classify_similarity(0.95) == ComparisonResult.SIMILAR
    assert engine._classify_similarity(0.85) == ComparisonResult.MINOR_CHANGES
    assert engine._classify_similarity(0.65) == ComparisonResult.MAJOR_CHANGES
    assert engine._classify_similarity(0.45) == ComparisonResult.REWRITTEN
    assert engine._classify_similarity(0.35) == ComparisonResult.MAJOR_CHANGES  # Below rewritten threshold


def test_comparison_with_error():
    """Test error handling in compare_sections method."""
    engine = EmbeddingComparisonEngine()

    # Mock _calculate_similarity to raise an exception
    engine._calculate_similarity = MagicMock(side_effect=ValueError("Test error"))

    with pytest.raises(DocumentProcessingError):
        engine.compare_sections(
            [
                AlignmentPair(
                    source_section=Section(id="s1", type=SectionType.PARAGRAPH, content="Source"),
                    target_section=Section(id="s2", type=SectionType.PARAGRAPH, content="Target"),
                )
            ]
        )
