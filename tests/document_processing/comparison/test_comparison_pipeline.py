"""Tests for the ComparisonPipeline class."""

from unittest.mock import MagicMock, patch

import pytest

from llm_rag.document_processing.comparison.domain_models import (
    AlignmentPair,
    ComparisonPipelineConfig,
    ComparisonResultType,
    DocumentFormat,
    Section,
    SectionComparison,
    SectionType,
)
from llm_rag.document_processing.comparison.pipeline import ComparisonPipeline
from llm_rag.utils.errors import DocumentProcessingError


@pytest.fixture
def mock_sections_cp():
    """Create mock sections for pipeline testing."""
    return [
        Section(title='Intro', content='Introduction', level=1, section_type=SectionType.HEADING),
        Section(title='P1', content='This is a test paragraph.', level=0, section_type=SectionType.PARAGRAPH),
    ]


@pytest.fixture
def mock_aligned_pairs_cp(mock_sections_cp):
    """Create mock alignment pairs for pipeline testing."""
    return [
        AlignmentPair(
            source_section=mock_sections_cp[0],
            target_section=mock_sections_cp[0],
            similarity_score=1.0,
        ),
        AlignmentPair(
            source_section=mock_sections_cp[1],
            target_section=mock_sections_cp[1],
            similarity_score=0.9,
        ),
    ]


@pytest.fixture
def mock_comparisons_cp(mock_aligned_pairs_cp):
    """Create mock comparison results for pipeline testing."""
    return [
        SectionComparison(
            alignment_pair=pair,
            result_type=ComparisonResultType.SIMILAR,
            similarity_score=pair.similarity_score,
        )
        for pair in mock_aligned_pairs_cp
    ]


@pytest.fixture
def mock_pipeline_cp():
    """Create a pipeline with mocked components for pipeline testing."""
    pipeline = ComparisonPipeline()
    pipeline.parser = MagicMock()
    pipeline.aligner = MagicMock()
    pipeline.comparison_engine = MagicMock()
    pipeline.formatter = MagicMock()
    return pipeline


def test_init_with_default_config_cp():
    """Test initialization with default configuration."""
    pipeline = ComparisonPipeline()
    assert isinstance(pipeline.config, ComparisonPipelineConfig)
    assert pipeline.config.cache_intermediate_results is False
    assert pipeline._cache == {}


def test_init_with_custom_config_cp():
    """Test initialization with custom configuration."""
    config = ComparisonPipelineConfig(cache_intermediate_results=True)
    pipeline = ComparisonPipeline(config)
    assert pipeline.config.cache_intermediate_results is True


def test_compare_documents_cp(mock_pipeline_cp, mock_sections_cp, mock_aligned_pairs_cp, mock_comparisons_cp):
    """Test the compare_documents method with mocked components."""
    mock_pipeline_cp.parser.parse.return_value = mock_sections_cp
    mock_pipeline_cp.aligner.align_sections.return_value = mock_aligned_pairs_cp
    mock_pipeline_cp.comparison_engine.compare_sections.return_value = mock_comparisons_cp
    mock_pipeline_cp.formatter.format_comparisons.return_value = 'Mock diff report'
    source = '# Introduction\nThis is a test paragraph.'
    target = '# Introduction\nThis is a test paragraph with minor changes.'
    result = mock_pipeline_cp.compare_documents(source, target, title='Test Diff')
    mock_pipeline_cp.parser.parse.assert_called()
    mock_pipeline_cp.aligner.align_sections.assert_called_once()
    mock_pipeline_cp.comparison_engine.compare_sections.assert_called_once()
    mock_pipeline_cp.formatter.format_comparisons.assert_called_once()
    assert result == 'Mock diff report'


def test_compare_sections_cp(mock_pipeline_cp, mock_sections_cp, mock_aligned_pairs_cp, mock_comparisons_cp):
    """Test the compare_sections method with mocked components."""
    mock_pipeline_cp.aligner.align_sections.return_value = mock_aligned_pairs_cp
    mock_pipeline_cp.comparison_engine.compare_sections.return_value = mock_comparisons_cp
    mock_pipeline_cp.formatter.format_comparisons.return_value = 'Mock diff report'
    result = mock_pipeline_cp.compare_sections(mock_sections_cp, mock_sections_cp, title='Test Diff')
    mock_pipeline_cp.aligner.align_sections.assert_called_once()
    mock_pipeline_cp.comparison_engine.compare_sections.assert_called_once()
    mock_pipeline_cp.formatter.format_comparisons.assert_called_once()
    assert result == 'Mock diff report'


def test_load_document_from_content():
    """Test loading document from content string."""
    pipeline = ComparisonPipeline()

    # Content string
    content = '# Heading\nThis is paragraph text.'

    result = pipeline._load_document(content)
    assert result == content


@patch('pathlib.Path.read_text')
def test_load_document_from_path(mock_read_text):
    """Test loading document from file path."""
    pipeline = ComparisonPipeline()

    # Mock Path.read_text to avoid file system access
    mock_read_text.return_value = '# Heading\nThis is paragraph text.'

    # Mock file path
    with patch('pathlib.Path.exists', return_value=True):
        result = pipeline._load_document('test_document.md')

    assert result == '# Heading\nThis is paragraph text.'
    mock_read_text.assert_called_once()


def test_load_document_error():
    """Test error handling in _load_document method."""
    pipeline = ComparisonPipeline()

    # Create a Path that will raise an error when read_text is called
    with patch('pathlib.Path.read_text', side_effect=PermissionError('Access denied')):
        with patch('pathlib.Path.exists', return_value=True):
            with pytest.raises(DocumentProcessingError):
                pipeline._load_document('nonexistent_file.md')


def test_parse_document_cp(mock_pipeline_cp, mock_sections_cp):
    """Test the _parse_document method."""
    mock_pipeline_cp.parser.parse.return_value = mock_sections_cp
    mock_pipeline_cp._load_document = MagicMock(return_value='# Heading\nParagraph text.')
    result = mock_pipeline_cp._parse_document('test_document.md', DocumentFormat.MARKDOWN)
    mock_pipeline_cp._load_document.assert_called_once()
    mock_pipeline_cp.parser.parse.assert_called_once()
    assert result == mock_sections_cp


def test_parse_document_with_cache_cp(mock_pipeline_cp, mock_sections_cp):
    """Test caching in the _parse_document method."""
    mock_pipeline_cp.parser.parse.return_value = mock_sections_cp
    mock_pipeline_cp._load_document = MagicMock(return_value='# Heading\nParagraph text.')
    mock_pipeline_cp.config.cache_intermediate_results = True
    result1 = mock_pipeline_cp._parse_document('test_document.md', DocumentFormat.MARKDOWN)
    result2 = mock_pipeline_cp._parse_document('test_document.md', DocumentFormat.MARKDOWN)
    assert mock_pipeline_cp._load_document.call_count == 1
    assert mock_pipeline_cp.parser.parse.call_count == 1
    assert result1 == mock_sections_cp
    assert result2 == mock_sections_cp
    assert 'parsed_test_document.md' in mock_pipeline_cp._cache


def test_align_sections_cp(mock_pipeline_cp, mock_sections_cp, mock_aligned_pairs_cp):
    """Test the _align_sections method."""
    mock_pipeline_cp.aligner.align_sections.return_value = mock_aligned_pairs_cp
    result = mock_pipeline_cp._align_sections(mock_sections_cp, mock_sections_cp)
    mock_pipeline_cp.aligner.align_sections.assert_called_once()
    assert result == mock_aligned_pairs_cp


def test_align_sections_with_cache_cp(mock_pipeline_cp, mock_sections_cp, mock_aligned_pairs_cp):
    """Test caching in the _align_sections method."""
    mock_pipeline_cp.aligner.align_sections.return_value = mock_aligned_pairs_cp
    mock_pipeline_cp.config.cache_intermediate_results = True
    result1 = mock_pipeline_cp._align_sections(mock_sections_cp, mock_sections_cp)
    result2 = mock_pipeline_cp._align_sections(mock_sections_cp, mock_sections_cp)
    assert mock_pipeline_cp.aligner.align_sections.call_count == 1
    assert result1 == mock_aligned_pairs_cp
    assert result2 == mock_aligned_pairs_cp
    cache_key = f'aligned_{id(mock_sections_cp)}_{id(mock_sections_cp)}'
    assert cache_key in mock_pipeline_cp._cache


def test_compare_sections_method_cp(mock_pipeline_cp, mock_aligned_pairs_cp, mock_comparisons_cp):
    """Test the _compare_sections method."""
    mock_pipeline_cp.comparison_engine.compare_sections.return_value = mock_comparisons_cp
    result = mock_pipeline_cp._compare_sections(mock_aligned_pairs_cp)
    mock_pipeline_cp.comparison_engine.compare_sections.assert_called_once()
    assert result == mock_comparisons_cp


def test_compare_sections_with_cache_cp(mock_pipeline_cp, mock_aligned_pairs_cp, mock_comparisons_cp):
    """Test caching in the _compare_sections method."""
    mock_pipeline_cp.comparison_engine.compare_sections.return_value = mock_comparisons_cp
    mock_pipeline_cp.config.cache_intermediate_results = True
    result1 = mock_pipeline_cp._compare_sections(mock_aligned_pairs_cp)
    result2 = mock_pipeline_cp._compare_sections(mock_aligned_pairs_cp)
    assert mock_pipeline_cp.comparison_engine.compare_sections.call_count == 1
    assert result1 == mock_comparisons_cp
    assert result2 == mock_comparisons_cp
    cache_key = f'compared_{id(mock_aligned_pairs_cp)}'
    assert cache_key in mock_pipeline_cp._cache


def test_format_results_cp(mock_pipeline_cp, mock_comparisons_cp):
    """Test the _format_results method."""
    mock_pipeline_cp.formatter.format_comparisons.return_value = 'Mock diff report'
    result = mock_pipeline_cp._format_results(mock_comparisons_cp, 'Test Diff')
    mock_pipeline_cp.formatter.format_comparisons.assert_called_once()
    assert result == 'Mock diff report'


def test_error_handling_in_compare_documents_cp(mock_pipeline_cp):
    """Test error handling in the compare_documents method."""
    mock_pipeline_cp.parser.parse.side_effect = ValueError('Parser error')
    with pytest.raises(DocumentProcessingError):
        mock_pipeline_cp.compare_documents('source', 'target')


def test_error_handling_in_compare_sections_cp(mock_pipeline_cp, mock_sections_cp):
    """Test error handling in the compare_sections method."""
    mock_pipeline_cp.aligner.align_sections.side_effect = ValueError('Aligner error')
    with pytest.raises(DocumentProcessingError):
        mock_pipeline_cp.compare_sections(mock_sections_cp, mock_sections_cp)


def test_end_to_end_with_real_components_cp():
    """Test pipeline with mocked components to simulate end-to-end workflow."""
    pipeline = ComparisonPipeline()
    pipeline.parser = MagicMock()
    pipeline.aligner = MagicMock()
    pipeline.comparison_engine = MagicMock()
    pipeline.formatter = MagicMock()
    pipeline.parser.parse.return_value = [
        Section(title='H1', content='Heading', level=1, section_type=SectionType.HEADING),
        Section(title='P1', content='Content', level=0, section_type=SectionType.PARAGRAPH),
    ]
    pipeline.aligner.align_sections.return_value = [
        AlignmentPair(
            source_section=Section(title='H1', content='Heading', level=1, section_type=SectionType.HEADING),
            target_section=Section(title='H1', content='Heading', level=1, section_type=SectionType.HEADING),
            similarity_score=1.0,
        )
    ]
    pipeline.comparison_engine.compare_sections.return_value = [
        SectionComparison(
            alignment_pair=AlignmentPair(
                source_section=Section(title='H1', content='Heading', level=1, section_type=SectionType.HEADING),
                target_section=Section(title='H1', content='Heading', level=1, section_type=SectionType.HEADING),
                similarity_score=1.0,
            ),
            result_type=ComparisonResultType.SIMILAR,
            similarity_score=1.0,
        )
    ]
    pipeline.formatter.format_comparisons.return_value = '# Comparison Result\n\nHeading [SIMILAR]'
    source = '# Heading\nThis is a paragraph.'
    target = '# Heading\nThis is a modified paragraph.'
    result = pipeline.compare_documents(source, target, title='Simple Test')
    assert isinstance(result, str)
    assert len(result) > 0
    assert 'Heading' in result
    pipeline.parser.parse.assert_called()
    pipeline.aligner.align_sections.assert_called_once()
    pipeline.comparison_engine.compare_sections.assert_called_once()
    pipeline.formatter.format_comparisons.assert_called_once()
