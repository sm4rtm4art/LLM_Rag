"""Tests for the EmbeddingComparisonEngine class."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from llm_rag.document_processing.comparison.comparison_engine import EmbeddingComparisonEngine
from llm_rag.document_processing.comparison.domain_models import (
    AlignmentPair as DomainAlignmentPair,
)
from llm_rag.document_processing.comparison.domain_models import (
    ComparisonConfig,
    ComparisonResultType,
    LLMAnalysisResult,
    Section,
    SectionType,
)
from llm_rag.document_processing.comparison.llm_comparer import LLMComparer
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


@pytest.mark.asyncio
async def test_compare_sections_ce(mock_alignment_pairs_ce):
    """Test comparing aligned section pairs."""
    engine = EmbeddingComparisonEngine()
    engine._calculate_similarity = MagicMock()
    engine._calculate_similarity.side_effect = [0.96, 0.75]  # Corresponds to 2 non-LLM pairs

    # Mock llm_comparer and its analyze_sections method if it might be called
    # For this test, assuming no LLM analysis is triggered by default or it's handled cleanly
    engine.llm_comparer = MagicMock(spec=LLMComparer)
    if engine.llm_comparer:
        engine.llm_comparer.analyze_sections = AsyncMock(
            return_value=LLMAnalysisResult(
                comparison_category='UNCERTAIN',  # Dummy value
                explanation='Test',
                confidence=0.5,
            )
        )

    comparisons = await engine.compare_sections(mock_alignment_pairs_ce)

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


@pytest.mark.asyncio
async def test_comparison_with_error_ce(mock_sections_ce):
    """Test error handling in compare_sections method."""
    engine = EmbeddingComparisonEngine()
    # Ensure the mock is set up to be an async method if llm_comparer might be called
    engine.llm_comparer = MagicMock(spec=LLMComparer)
    if engine.llm_comparer:
        engine.llm_comparer.analyze_sections = AsyncMock(side_effect=ValueError('LLM Test error'))

    engine._calculate_similarity = MagicMock(side_effect=ValueError('Similarity calculation error'))
    with pytest.raises(DocumentProcessingError):
        await engine.compare_sections(
            [
                DomainAlignmentPair(
                    source_section=mock_sections_ce[0],
                    target_section=mock_sections_ce[1],
                )
            ]
        )


# Directly access the constant from the module it's defined in for consistency
# If it's not meant to be public, tests might need to redefine it or use a test-specific value.
# For now, assuming it can be imported or re-defined if necessary for test setup.
# from llm_rag.document_processing.comparison.comparison_engine import MIN_LLM_CONTENT_LENGTH
# If the above import causes issues (e.g. circular), we can redefine it in tests:
MIN_LLM_TEST_CONTENT_LENGTH = 15

LONG_TEXT_SAMPLE = 'This is a sufficiently long text for LLM analysis, it has many words.'
SHORT_TEXT_SAMPLE = 'Too short'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'text_a, text_b, should_skip, expected_category',
    [
        (SHORT_TEXT_SAMPLE, LONG_TEXT_SAMPLE, True, 'NO_MEANINGFUL_CONTENT'),
        (LONG_TEXT_SAMPLE, SHORT_TEXT_SAMPLE, True, 'NO_MEANINGFUL_CONTENT'),
        (SHORT_TEXT_SAMPLE, SHORT_TEXT_SAMPLE, True, 'NO_MEANINGFUL_CONTENT'),
        (
            LONG_TEXT_SAMPLE,
            LONG_TEXT_SAMPLE,
            False,
            'UNCERTAIN',  # Example: if LLM was called
        ),
    ],
)
async def test_run_llm_analysis_for_pair_short_content_skip(text_a, text_b, should_skip, expected_category):
    """Test _run_llm_analysis_for_pair short content skipping logic."""
    mock_llm_comparer = MagicMock(spec=LLMComparer)
    # Setup the mock for analyze_sections to be an async method
    mock_llm_comparer.analyze_sections = AsyncMock(
        return_value=LLMAnalysisResult(
            comparison_category='UNCERTAIN',  # Default for non-skip case
            explanation='LLM was called',
            confidence=0.5,
        )
    )

    engine = EmbeddingComparisonEngine(llm_comparer=mock_llm_comparer)

    # Access the normally private method for testing - this is a common practice
    # for unit testing specific pieces of logic.
    # Ensure MIN_LLM_CONTENT_LENGTH used in the engine is consistent with test value
    # If engine.MIN_LLM_CONTENT_LENGTH is accessible, use that, otherwise trust the hardcoded one is synced.
    # For this test, we rely on the value used in the implementation, assumed to be 15.

    original_index = 0
    index, result = await engine._run_llm_analysis_for_pair(text_a, text_b, original_index)

    assert index == original_index
    assert result is not None
    assert result.comparison_category == expected_category

    if should_skip:
        assert result.explanation == ('One or both sections were too short for LLM analysis.')
        mock_llm_comparer.analyze_sections.assert_not_called()
    else:
        # This part tests that if content is NOT short, the LLM comparer IS called.
        mock_llm_comparer.analyze_sections.assert_called_once_with(text_a, text_b)
        assert result.explanation == 'LLM was called'  # From the mock return value
