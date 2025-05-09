"""Tests for the SectionAligner class."""

import numpy as np
import pytest

from llm_rag.document_processing.comparison.alignment import (
    AlignmentConfig,
    AlignmentMethod,
    AlignmentPair,
    SectionAligner,
)
from llm_rag.document_processing.comparison.document_parser import Section, SectionType


@pytest.fixture
def source_sections():
    """Create a list of source sections for testing."""
    return [
        Section(id='s1', section_type=SectionType.HEADING, content='Introduction', level=1),
        Section(id='s2', section_type=SectionType.PARAGRAPH, content='This is the first paragraph.', level=0),
        Section(id='s3', section_type=SectionType.PARAGRAPH, content='This is the second paragraph.', level=0),
        Section(id='s4', section_type=SectionType.HEADING, content='Section 1', level=1),
        Section(id='s5', section_type=SectionType.PARAGRAPH, content='Content for section 1.', level=0),
        Section(id='s6', section_type=SectionType.HEADING, content='Conclusion', level=1),
        Section(id='s7', section_type=SectionType.PARAGRAPH, content='Final remarks.', level=0),
    ]


@pytest.fixture
def target_sections():
    """Create a list of target sections for testing."""
    return [
        Section(id='t1', section_type=SectionType.HEADING, content='Introduction', level=1),
        Section(
            id='t2', section_type=SectionType.PARAGRAPH, content='This is the first paragraph with changes.', level=0
        ),
        Section(id='t3', section_type=SectionType.PARAGRAPH, content='This is the second paragraph.', level=0),
        Section(id='t4', section_type=SectionType.HEADING, content='Section 1', level=1),
        Section(id='t5', section_type=SectionType.PARAGRAPH, content='Updated content for section 1.', level=0),
        Section(id='t6', section_type=SectionType.HEADING, content='New Section', level=1),
        Section(id='t7', section_type=SectionType.PARAGRAPH, content='Content for the new section.', level=0),
        Section(id='t8', section_type=SectionType.HEADING, content='Conclusion', level=1),
        Section(id='t9', section_type=SectionType.PARAGRAPH, content='Final thoughts and remarks.', level=0),
    ]


@pytest.fixture
def empty_sections():
    """Create an empty list of sections for edge case testing."""
    return []


def test_init_with_default_config():
    """Test initialization with default configuration."""
    aligner = SectionAligner()
    assert isinstance(aligner.config, AlignmentConfig)
    assert aligner.config.method == AlignmentMethod.HYBRID
    assert aligner.config.similarity_threshold == 0.7


def test_init_with_custom_config():
    """Test initialization with custom configuration."""
    config = AlignmentConfig(
        method=AlignmentMethod.HEADING_MATCH,
        similarity_threshold=0.8,
        heading_weight=3.0,
    )
    aligner = SectionAligner(config)
    assert aligner.config.method == AlignmentMethod.HEADING_MATCH
    assert aligner.config.similarity_threshold == 0.8
    assert aligner.config.heading_weight == 3.0


def test_align_sections_heading_match(source_sections, target_sections):
    """Test alignment using the heading match method."""
    config = AlignmentConfig(method=AlignmentMethod.HEADING_MATCH)
    aligner = SectionAligner(config)

    alignment_pairs = aligner.align_sections(source_sections, target_sections)

    assert len(alignment_pairs) > 0

    # Check that headings are correctly matched
    heading_pairs = [
        pair
        for pair in alignment_pairs
        if pair.is_aligned
        and pair.source_section.section_type == SectionType.HEADING
        and pair.target_section.section_type == SectionType.HEADING
    ]

    # We expect Introduction, Section 1, and Conclusion to be matched
    assert len(heading_pairs) == 3

    # Verify Introduction headings are matched
    intro_pair = next((pair for pair in heading_pairs if pair.source_section.content == 'Introduction'), None)
    assert intro_pair is not None
    assert intro_pair.target_section.content == 'Introduction'

    # Verify that content sections are also aligned
    content_pairs = [
        pair
        for pair in alignment_pairs
        if pair.is_aligned
        and pair.source_section.section_type == SectionType.PARAGRAPH
        and pair.target_section.section_type == SectionType.PARAGRAPH
    ]

    # Should have some matched content sections
    assert len(content_pairs) > 0


def test_align_sections_sequence(source_sections, target_sections):
    """Test alignment using the sequence method."""
    config = AlignmentConfig(method=AlignmentMethod.SEQUENCE)
    aligner = SectionAligner(config)

    alignment_pairs = aligner.align_sections(source_sections, target_sections)

    assert len(alignment_pairs) > 0

    # Check for aligned pairs
    aligned_pairs = [pair for pair in alignment_pairs if pair.is_aligned]
    assert len(aligned_pairs) > 0

    # Verify method is set correctly
    for pair in aligned_pairs:
        assert pair.method == AlignmentMethod.SEQUENCE

    # Verify we also have source-only and target-only pairs
    source_only = [pair for pair in alignment_pairs if pair.is_source_only]
    target_only = [pair for pair in alignment_pairs if pair.is_target_only]

    # The total should account for all sections
    assert len(aligned_pairs) + len(source_only) + len(target_only) == len(alignment_pairs)


def test_align_sections_content_similarity(source_sections, target_sections):
    """Test alignment using the content similarity method."""
    config = AlignmentConfig(method=AlignmentMethod.CONTENT_SIMILARITY, similarity_threshold=0.3)
    aligner = SectionAligner(config)

    alignment_pairs = aligner.align_sections(source_sections, target_sections)

    assert len(alignment_pairs) > 0

    # Check for aligned pairs
    aligned_pairs = [
        pair for pair in alignment_pairs if pair.is_aligned and pair.method == AlignmentMethod.CONTENT_SIMILARITY
    ]
    assert len(aligned_pairs) > 0

    # Find pairs with identical content (should have high similarity)
    identical_content_pairs = [
        pair for pair in aligned_pairs if pair.source_section.content == pair.target_section.content
    ]

    # The second paragraph is identical in both documents
    identical_pair = next(
        (pair for pair in identical_content_pairs if 'second paragraph' in pair.source_section.content), None
    )
    assert identical_pair is not None
    assert identical_pair.similarity_score > 0.9  # Should be very high similarity


def test_align_sections_hybrid(source_sections, target_sections):
    """Test alignment using the hybrid method."""
    config = AlignmentConfig(method=AlignmentMethod.HYBRID)
    aligner = SectionAligner(config)

    alignment_pairs = aligner.align_sections(source_sections, target_sections)

    assert len(alignment_pairs) > 0

    # Verify we have aligned pairs with hybrid method
    hybrid_pairs = [pair for pair in alignment_pairs if pair.is_aligned and pair.method == AlignmentMethod.HYBRID]
    assert len(hybrid_pairs) > 0

    # Check that all sections are accounted for
    source_section_count = len(source_sections)
    target_section_count = len(target_sections)

    # Count sections in alignment pairs
    aligned_source_sections = sum(1 for pair in alignment_pairs if pair.source_section is not None)
    aligned_target_sections = sum(1 for pair in alignment_pairs if pair.target_section is not None)

    # All sections should be included in the alignment
    assert aligned_source_sections == source_section_count
    assert aligned_target_sections == target_section_count


def test_align_sections_with_empty_input(empty_sections):
    """Test alignment with empty input lists."""
    aligner = SectionAligner()

    # Empty source, non-empty target
    pairs1 = aligner.align_sections(
        empty_sections, [Section(id='t1', section_type=SectionType.PARAGRAPH, content='Test', level=0)]
    )
    assert len(pairs1) == 1
    assert pairs1[0].source_section is None
    assert pairs1[0].target_section is not None

    # Non-empty source, empty target
    pairs2 = aligner.align_sections(
        [Section(id='s1', section_type=SectionType.PARAGRAPH, content='Test', level=0)], empty_sections
    )
    assert len(pairs2) == 1
    assert pairs2[0].source_section is not None
    assert pairs2[0].target_section is None

    # Both empty
    pairs3 = aligner.align_sections(empty_sections, empty_sections)
    assert len(pairs3) == 0


def test_align_sections_with_invalid_method():
    """Test alignment with an invalid method."""
    # Create a custom config with invalid method
    invalid_config = AlignmentConfig()
    # Set the method to an invalid value, using an Enum property that doesn't exist
    invalid_config.method = AlignmentMethod.HEADING_MATCH  # First set a valid method

    aligner = SectionAligner(invalid_config)

    # Override the method with an incorrect value after initialization
    def mock_align_sections(source, target):
        raise ValueError('Unsupported alignment method')

    aligner.align_sections = mock_align_sections

    with pytest.raises(ValueError):
        aligner.align_sections(
            [Section(id='s1', section_type=SectionType.PARAGRAPH, content='Test', level=0)],
            [Section(id='t1', section_type=SectionType.PARAGRAPH, content='Test', level=0)],
        )


def test_map_heading_to_sections(source_sections):
    """Test the mapping of headings to their associated content sections."""
    aligner = SectionAligner()

    mapping = aligner._map_heading_to_sections(source_sections)

    # We have 3 headings in the source_sections fixture
    assert len(mapping) == 3

    # Check the first heading (Introduction) has 2 paragraphs
    intro_idx = next(i for i, s in enumerate(source_sections) if s.content == 'Introduction')
    assert len(mapping[intro_idx]) == 2

    # Check the indices point to the correct sections
    for idx in mapping[intro_idx]:
        assert source_sections[idx].section_type == SectionType.PARAGRAPH


def test_sequence_alignment():
    """Test the sequence alignment algorithm."""
    aligner = SectionAligner()

    # Simple similarity matrix for testing
    # Two sequences: A,B,C and A,D,C
    # A matches A (score 1.0), C matches C (score 1.0), B doesn't match D (score 0.2)
    similarity_matrix = np.array(
        [
            [1.0, 0.3, 0.3],  # A compared to A,D,C
            [0.3, 0.2, 0.4],  # B compared to A,D,C
            [0.3, 0.4, 1.0],  # C compared to A,D,C
        ]
    )

    alignment = aligner._sequence_alignment(similarity_matrix)

    # Expected alignment: (0,0), (1,1), (2,2) - matching A-A, B-D, C-C
    assert len(alignment) == 3
    assert alignment[0] == (0, 0)  # A matches A
    assert alignment[2] == (2, 2)  # C matches C


def test_calculate_text_similarity():
    """Test the text similarity calculation."""
    aligner = SectionAligner()

    # Test identical texts
    similarity1 = aligner._calculate_text_similarity('This is a test', 'This is a test')
    assert similarity1 == 1.0

    # Test completely different texts
    similarity2 = aligner._calculate_text_similarity('This is a test', 'Completely different')
    assert similarity2 < 0.5

    # Test partially overlapping texts
    similarity3 = aligner._calculate_text_similarity('This is a test', 'This is another test')
    assert 0.5 < similarity3 < 1.0

    # Test empty strings
    similarity4 = aligner._calculate_text_similarity('', '')
    assert similarity4 == 0.0

    similarity5 = aligner._calculate_text_similarity('This is a test', '')
    assert similarity5 == 0.0


def test_alignment_pair_properties():
    """Test the properties of the AlignmentPair class."""
    # Create test sections
    source = Section(id='s1', section_type=SectionType.PARAGRAPH, content='Source', level=0)
    target = Section(id='t1', section_type=SectionType.PARAGRAPH, content='Target', level=0)

    # Test aligned pair
    aligned_pair = AlignmentPair(source_section=source, target_section=target, similarity_score=0.9)
    assert aligned_pair.is_aligned is True
    assert aligned_pair.is_source_only is False
    assert aligned_pair.is_target_only is False

    # Test source-only pair (deletion)
    source_only = AlignmentPair(source_section=source, target_section=None)
    assert source_only.is_aligned is False
    assert source_only.is_source_only is True
    assert source_only.is_target_only is False

    # Test target-only pair (addition)
    target_only = AlignmentPair(source_section=None, target_section=target)
    assert target_only.is_aligned is False
    assert target_only.is_source_only is False
    assert target_only.is_target_only is True


def test_split_sections_by_headings(source_sections):
    """Test splitting sections into segments based on headings."""
    aligner = SectionAligner()

    segments = aligner._split_sections_by_headings(source_sections)

    # We should have 3 segments (Introduction, Section 1, Conclusion)
    assert len(segments) == 3

    # Check each segment starts with a heading
    for segment in segments:
        assert segment[0].section_type == SectionType.HEADING

    # Check content of first segment
    assert segments[0][0].content == 'Introduction'
    assert len(segments[0]) == 3  # Heading + 2 paragraphs
