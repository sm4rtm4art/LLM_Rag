import html
from typing import List

import pytest

from llm_rag.document_processing.comparison.comparison_engine import (
    AlignmentPair,
    ComparisonResult,
    SectionComparison,
)
from llm_rag.document_processing.comparison.diff_formatter import (
    AnnotationStyle,
    DiffFormat,
    DiffFormatter,
    FormatterConfig,
)
from llm_rag.document_processing.comparison.document_parser import Section, SectionType


@pytest.fixture
def default_config() -> FormatterConfig:
    """Returns a default FormatterConfig."""
    return FormatterConfig()


@pytest.fixture
def mock_section_comparison() -> SectionComparison:
    """Returns a mock SectionComparison object for testing."""
    source_section = Section(id='doc1-s1', content='This is the original text.', section_type=SectionType.PARAGRAPH)
    target_section = Section(id='doc2-t1', content='This is the modified text.', section_type=SectionType.PARAGRAPH)
    alignment_pair = AlignmentPair(source_section=source_section, target_section=target_section)
    return SectionComparison(
        alignment_pair=alignment_pair,
        result=ComparisonResult.MAJOR_CHANGES,
        similarity_score=0.75,
        details={'explanation': 'Texts differ significantly.'},
    )


@pytest.fixture
def mock_section_comparison_similar() -> SectionComparison:
    """Returns a mock SectionComparison object representing similar sections."""
    source_section = Section(id='doc1-s1-similar', content='This is some text.', section_type=SectionType.PARAGRAPH)
    target_section = Section(id='doc2-t1-similar', content='This is some text.', section_type=SectionType.PARAGRAPH)
    alignment_pair = AlignmentPair(source_section=source_section, target_section=target_section)
    return SectionComparison(
        alignment_pair=alignment_pair,
        result=ComparisonResult.SIMILAR,
        similarity_score=1.0,
        details={'explanation': 'Texts are identical.'},
    )


@pytest.fixture
def mock_section_comparison_new() -> SectionComparison:
    """Returns a mock SectionComparison object for a new section."""
    source_section = Section(id='doc1-s_new_placeholder', content='', section_type=SectionType.UNKNOWN)
    target_section = Section(
        id='doc2-t_new', content='This is a brand new section.', section_type=SectionType.PARAGRAPH
    )
    alignment_pair = AlignmentPair(source_section=source_section, target_section=target_section)
    return SectionComparison(
        alignment_pair=alignment_pair,
        result=ComparisonResult.NEW,
        similarity_score=0.0,
        details={'explanation': 'New section in target.'},
    )


@pytest.fixture
def mock_section_comparison_deleted() -> SectionComparison:
    """Returns a mock SectionComparison object for a deleted section."""
    source_section = Section(
        id='doc1-s_deleted', content='This section will be deleted.', section_type=SectionType.PARAGRAPH
    )
    target_section = Section(id='doc2-t_del_placeholder', content='', section_type=SectionType.UNKNOWN)
    alignment_pair = AlignmentPair(source_section=source_section, target_section=target_section)
    return SectionComparison(
        alignment_pair=alignment_pair,
        result=ComparisonResult.DELETED,
        similarity_score=0.0,
        details={'explanation': 'Section deleted from source.'},
    )


class TestDiffFormatter:
    """Tests for the DiffFormatter class."""

    def test_formatter_initialization_default_config(self):
        """Test DiffFormatter initializes with default configuration."""
        formatter = DiffFormatter()
        assert formatter.config.output_format == DiffFormat.MARKDOWN
        assert formatter.config.annotation_style == AnnotationStyle.STANDARD
        assert formatter.config.include_similarity_scores is True
        assert formatter.config.show_unchanged is True

    def test_formatter_initialization_custom_config(self):
        """Test DiffFormatter initializes with custom configuration."""
        custom_config = FormatterConfig(
            output_format=DiffFormat.HTML,
            annotation_style=AnnotationStyle.DETAILED,
            include_similarity_scores=False,
            show_unchanged=False,
        )
        formatter = DiffFormatter(config=custom_config)
        assert formatter.config.output_format == DiffFormat.HTML
        assert formatter.config.annotation_style == AnnotationStyle.DETAILED
        assert formatter.config.include_similarity_scores is False
        assert formatter.config.show_unchanged is False

    def test_format_comparisons_empty_list(self, default_config: FormatterConfig):
        """Test formatting an empty list of comparisons."""
        formatter = DiffFormatter(config=default_config)
        comparisons: List[SectionComparison] = []
        report = formatter.format_comparisons(comparisons, title='Empty Report')
        assert 'Empty Report' in report
        # Further checks depending on the default format (e.g., Markdown summary)
        assert 'Similar sections: 0' in report
        assert 'New: 0' in report  # Check for summary elements

    def test_format_markdown_basic(self, mock_section_comparison: SectionComparison):
        """Test basic Markdown formatting for a changed section."""
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison], title='Markdown Test')

        assert '# Markdown Test' in report
        assert '## Summary' in report
        assert (
            f'- {ComparisonResult.MAJOR_CHANGES.value.replace("_", " ").title().replace("Changes", "changes")}: 1'
            in report
        )
        assert '## Details' in report
        assert f'### Section 1: {mock_section_comparison.result.value.upper()}' in report
        assert f'*Similarity: {mock_section_comparison.similarity_score:.2f}*' in report
        assert 'Source:' in report
        assert '```' in report
        assert mock_section_comparison.alignment_pair.source_section.content in report
        assert 'Target:' in report
        assert mock_section_comparison.alignment_pair.target_section.content in report
        assert '---' in report  # Separator

    def test_format_markdown_new_section(self, mock_section_comparison_new: SectionComparison):
        """Test Markdown formatting for a new section."""
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_new], title='New Section MD Test')
        assert f'### Section 1: {ComparisonResult.NEW.value.upper()}' in report
        assert '**[NEW]**' in report
        assert mock_section_comparison_new.alignment_pair.target_section.content in report
        assert 'Source:' not in report

    def test_format_markdown_deleted_section(self, mock_section_comparison_deleted: SectionComparison):
        """Test Markdown formatting for a deleted section."""
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_deleted], title='Deleted Section MD Test')
        assert f'### Section 1: {ComparisonResult.DELETED.value.upper()}' in report
        assert '**[DELETED]**' in report
        assert mock_section_comparison_deleted.alignment_pair.source_section.content in report
        assert 'Target:' not in report

    def test_format_markdown_similar_section_show_unchanged_true(
        self, mock_section_comparison_similar: SectionComparison
    ):
        """Test Markdown formatting for a similar section when show_unchanged is True."""
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN, show_unchanged=True)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_similar])
        assert f'### Section 1: {ComparisonResult.SIMILAR.value.upper()}' in report
        assert '**[SIMILAR]**' in report
        assert mock_section_comparison_similar.alignment_pair.source_section.content in report

    def test_format_markdown_similar_section_show_unchanged_false(
        self, mock_section_comparison_similar: SectionComparison
    ):
        """Test Markdown formatting for a similar section when show_unchanged is False."""
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN, show_unchanged=False)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_similar])
        assert f'### Section 1: {ComparisonResult.SIMILAR.value.upper()}' not in report
        assert '## Summary' in report
        assert '- Similar sections: 1' in report  # Based on actual output format

    def test_format_markdown_detailed_annotation(self, mock_section_comparison: SectionComparison):
        """Test Markdown formatting with detailed annotation style for changed sections."""
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN, annotation_style=AnnotationStyle.DETAILED)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison])
        assert 'Diff:' in report
        assert '```diff' in report
        assert '+This is the modified text.' in report or '-This is the original text.' in report

    def test_format_html_basic(self, mock_section_comparison: SectionComparison):
        """Test basic HTML formatting for a changed section."""
        config = FormatterConfig(output_format=DiffFormat.HTML)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison], title='HTML Test')

        assert '<title>HTML Test</title>' in report
        assert "<div class='summary'>" in report
        assert (
            '<li>{}: 1</li>'.format(
                ComparisonResult.MAJOR_CHANGES.value.replace('_', ' ').title().replace('Changes', 'changes')
            )
            in report
        )
        assert '<h2>Details</h2>' in report
        assert f"<div class='section {mock_section_comparison.result.value.lower()}'>" in report
        assert f"<div class='header'>Section 1: {mock_section_comparison.result.value.upper()}</div>" in report
        assert f"<div class='similarity'>Similarity: {mock_section_comparison.similarity_score:.2f}</div>" in report
        assert "<div class='source'>" in report
        assert '<strong>Source:</strong>' in report
        assert self._escape_html_for_test(mock_section_comparison.alignment_pair.source_section.content) in report
        assert "<div class='target'>" in report
        assert '<strong>Target:</strong>' in report
        assert self._escape_html_for_test(mock_section_comparison.alignment_pair.target_section.content) in report

    def test_format_html_detailed_annotation(self, mock_section_comparison: SectionComparison):
        """Test HTML formatting with detailed annotation style."""
        config = FormatterConfig(output_format=DiffFormat.HTML, annotation_style=AnnotationStyle.DETAILED)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison])
        assert "<div class='diff'>" in report
        assert '<strong>Diff:</strong>' in report
        assert "<div class='diff-add'>" in report or "<div class='diff-remove'>" in report

    def test_format_text_basic(self, mock_section_comparison: SectionComparison):
        """Test basic Text formatting for a changed section."""
        config = FormatterConfig(output_format=DiffFormat.TEXT)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison], title='Text Test')

        assert 'TEXT TEST' in report
        assert 'SUMMARY' in report
        assert (
            f'{ComparisonResult.MAJOR_CHANGES.value.replace("_", " ").title().replace("Changes", "changes")}: 1'
            in report
        )
        assert 'DETAILS' in report
        assert f'Section 1: {mock_section_comparison.result.value.upper()}' in report
        assert f'Similarity: {mock_section_comparison.similarity_score:.2f}' in report
        assert 'Source:' in report
        assert mock_section_comparison.alignment_pair.source_section.content in report  # Plain text
        assert 'Target:' in report
        assert mock_section_comparison.alignment_pair.target_section.content in report  # Plain text
        assert '====' in report  # Separator

    def test_format_text_detailed_annotation(self, mock_section_comparison: SectionComparison):
        """Test Text formatting with detailed annotation style."""
        config = FormatterConfig(output_format=DiffFormat.TEXT, annotation_style=AnnotationStyle.DETAILED)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison])
        assert 'Diff:' in report
        assert '+This is the modified text.' in report or '-This is the original text.' in report

    def test_generate_text_diff(self, default_config: FormatterConfig):
        """Test the _generate_text_diff method."""
        formatter = DiffFormatter(config=default_config)  # Config not directly used by this method but good practice
        text1 = 'line one\nline two\nline three'
        text2 = 'line one\nline two modified\nline three'
        diff_output = formatter._generate_text_diff(text1, text2)
        assert '--- source' in diff_output
        assert '+++ target' in diff_output
        assert '-line two' in diff_output
        assert '+line two modified' in diff_output
        assert ' line one' in diff_output  # Context line

    def test_generate_html_diff(self, default_config: FormatterConfig):
        """Test the _generate_html_diff method."""
        formatter = DiffFormatter(config=default_config)
        text1 = 'line one\nline two'  # Changed from \\n
        text2 = 'line one\nline new\nline two'  # Changed from \\n
        html_diff = formatter._generate_html_diff(text1, text2)
        # Expected output based on difflib.Differ logic:
        expected_parts = ['<div>line one</div>', "<div class='diff-add'>line new</div>", '<div>line two</div>']
        actual_parts = [part.strip() for part in html_diff.splitlines() if part.strip()]
        assert actual_parts == expected_parts
        # Keep a simpler check as well, in case join order or extra newlines are an issue
        assert "<div class='diff-add'>line new</div>" in html_diff

    def test_escape_html(self, default_config: FormatterConfig):
        """Test the _escape_html utility method."""
        formatter = DiffFormatter(config=default_config)
        input_text = '<tag>&\'"'  # Note: single backslash for the quote
        expected_escaped_text = html.escape(input_text)
        assert formatter._escape_html(input_text) == expected_escaped_text

    def test_wrap_text(self):
        """Test the _wrap_text utility method."""
        config = FormatterConfig(wrap_width=10)
        formatter = DiffFormatter(config=config)
        text = 'This is a long line of text.'
        wrapped = formatter._wrap_text(text)
        expected = 'This is a\nlong line\nof text.'
        assert wrapped == expected

        config_no_wrap = FormatterConfig(wrap_width=0)  # Or some indicator for no wrap
        formatter_no_wrap = DiffFormatter(config=config_no_wrap)
        assert formatter_no_wrap._wrap_text(text) == text

    # Helper method for HTML tests to avoid repeating escape logic
    def _escape_html_for_test(self, text: str) -> str:
        return html.escape(text)

    # TODO:
    # - Test different AnnotationStyle values more thoroughly (MINIMAL).
    # - Test FormatterConfig: include_metadata.
    # - Test FormatterConfig: color_output (might be hard to assert directly,
    #   maybe check if specific ANSI codes are present/absent).
    # - Test edge cases: e.g., completely empty content in sections.
    # - Test error handling if an unsupported format is somehow passed (though enum should prevent this).
    # - Test formatting of all ComparisonResult types for each output format.
    # - Test multiple comparisons in one report.
