"""Tests for the EmbeddingComparisonEngine class."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from llm_rag.document_processing.comparison.comparison_engine import EmbeddingComparisonEngine
from llm_rag.document_processing.comparison.domain_models import AlignmentPair as DomainAlignmentPair
from llm_rag.document_processing.comparison.domain_models import (
    ComparisonConfig,
    ComparisonResultType,
    Section,
    SectionType,
)
from llm_rag.utils.errors import DocumentProcessingError


@pytest.fixture
def mock_sections_ce():
    """Create mock sections for comparison engine testing."""
    return [
        Section(title='Intro', content='Introduction', level=1, section_type=SectionType.HEADING),
        Section(
            title='P1',
            content='This is a test paragraph with some content.',
            level=0,
            section_type=SectionType.PARAGRAPH,
        ),
        Section(
            title='P2',
            content='This paragraph has been modified with different text.',
            level=0,
            section_type=SectionType.PARAGRAPH,
        ),
        Section(title='Conclusion', content='Conclusion', level=1, section_type=SectionType.HEADING),
        Section(title='P3', content='Final remarks go here.', level=0, section_type=SectionType.PARAGRAPH),
    ]


@pytest.fixture
def mock_alignment_pairs_ce(mock_sections_ce):
    """Create mock alignment pairs for comparison engine testing."""
    return [
        DomainAlignmentPair(
            source_section=mock_sections_ce[0],
            target_section=mock_sections_ce[0],
        ),
        DomainAlignmentPair(
            source_section=mock_sections_ce[1],
            target_section=mock_sections_ce[2],
        ),
        DomainAlignmentPair(
            source_section=mock_sections_ce[3],
            target_section=None,
        ),
        DomainAlignmentPair(
            source_section=None,
            target_section=mock_sections_ce[4],
        ),
    ]


def test_init_with_default_config_ce():
    """Test initialization with default configuration."""
    engine = EmbeddingComparisonEngine()
    assert isinstance(engine.config, ComparisonConfig)
    assert engine.config.similarity_thresholds.similar == 0.95
    assert engine._embedding_model is None
    assert isinstance(engine._embedding_cache, dict)


def test_init_with_custom_config_ce():
    """Test initialization with custom configuration."""
    from llm_rag.document_processing.comparison.domain_models import SimilarityThresholds

    custom_thresholds = SimilarityThresholds(similar=0.95, modified=0.85, different=0.7)
    config = ComparisonConfig(
        similarity_thresholds=custom_thresholds,
        embedding_model_name='custom-model',
    )
    engine = EmbeddingComparisonEngine(config)
    assert engine.config.similarity_thresholds.similar == 0.95
    assert engine.config.similarity_thresholds.modified == 0.85
    assert engine.config.embedding_model_name == 'custom-model'


def test_compare_sections_ce(mock_alignment_pairs_ce):
    """Test comparing aligned section pairs."""
    engine = EmbeddingComparisonEngine()
    engine._calculate_similarity = MagicMock()
    engine._calculate_similarity.side_effect = [0.96, 0.75]

    comparisons = engine.compare_sections(mock_alignment_pairs_ce)

    assert len(comparisons) == 4
    assert comparisons[0].result_type == ComparisonResultType.SIMILAR
    assert comparisons[1].result_type == ComparisonResultType.DIFFERENT
    assert comparisons[2].result_type == ComparisonResultType.DELETED
    assert comparisons[3].result_type == ComparisonResultType.NEW
    assert engine._calculate_similarity.call_count == 2


def test_calculate_similarity_ce():
    """Test calculating similarity between two sections."""
    engine = EmbeddingComparisonEngine()
    engine._get_embedding = MagicMock()
    engine._get_embedding.side_effect = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.707, 0.707, 0.0]),
    ]
    source = Section(title='S1', content='Source text', level=0, section_type=SectionType.PARAGRAPH)
    target = Section(title='T1', content='Target text', level=0, section_type=SectionType.PARAGRAPH)
    similarity = engine._calculate_similarity(source, target)
    assert 0.7 <= similarity <= 0.72
    assert engine._get_embedding.call_count == 2


def test_get_embedding_cache_ce():
    """Test embedding caching mechanism."""
    engine = EmbeddingComparisonEngine()
    engine._compute_embedding = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))
    result1 = engine._get_embedding('test text')
    assert engine._compute_embedding.call_count == 1
    result2 = engine._get_embedding('test text')
    assert engine._compute_embedding.call_count == 1
    assert np.array_equal(result1, result2)
    engine._get_embedding('different text')
    assert engine._compute_embedding.call_count == 2


def test_compute_embedding_ce():
    """Test computing embeddings for text."""
    engine = EmbeddingComparisonEngine()
    engine._initialize_embedding_model()
    embedding1 = engine._compute_embedding('Simple test text')
    assert isinstance(embedding1, np.ndarray)
    assert embedding1.shape[0] > 0
    embedding2 = engine._compute_embedding('Simple test text')
    assert np.array_equal(embedding1, embedding2)
    embedding3 = engine._compute_embedding('Different text')
    assert not np.array_equal(embedding1, embedding3)


def test_classify_similarity_ce():
    """Test classification of similarity scores."""
    engine = EmbeddingComparisonEngine()
    assert engine._classify_similarity(0.96) == ComparisonResultType.SIMILAR
    assert engine._classify_similarity(0.95) == ComparisonResultType.SIMILAR
    assert engine._classify_similarity(0.85) == ComparisonResultType.MODIFIED
    assert engine._classify_similarity(0.80) == ComparisonResultType.MODIFIED
    assert engine._classify_similarity(0.70) == ComparisonResultType.DIFFERENT
    assert engine._classify_similarity(0.60) == ComparisonResultType.DIFFERENT
    assert engine._classify_similarity(0.45) == ComparisonResultType.DIFFERENT
    assert engine._classify_similarity(0.35) == ComparisonResultType.DIFFERENT


def test_comparison_with_error_ce(mock_sections_ce):
    """Test error handling in compare_sections method."""
    engine = EmbeddingComparisonEngine()
    engine._calculate_similarity = MagicMock(side_effect=ValueError('Test error'))
    with pytest.raises(DocumentProcessingError):
        engine.compare_sections(
            [
                DomainAlignmentPair(
                    source_section=mock_sections_ce[0],
                    target_section=mock_sections_ce[1],
                )
            ]
        )
