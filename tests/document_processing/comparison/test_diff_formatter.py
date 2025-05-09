import html
from typing import List

import pytest

from llm_rag.document_processing.comparison.diff_formatter import DiffFormatter
from llm_rag.document_processing.comparison.domain_models import (
    AlignmentPair,
    AnnotationStyle,
    ComparisonResultType,
    DiffFormat,
    FormatterConfig,
    Section,
    SectionComparison,
    SectionType,
)


@pytest.fixture
def default_config_df():
    """Returns a default FormatterConfig."""
    return FormatterConfig()


@pytest.fixture
def mock_section_comparison_df():
    """Returns a mock SectionComparison object for testing."""
    source_section = Section(
        title='S1', content='This is the original text.', level=0, section_type=SectionType.PARAGRAPH
    )
    target_section = Section(
        title='T1', content='This is the modified text.', level=0, section_type=SectionType.PARAGRAPH
    )
    alignment_pair = AlignmentPair(source_section=source_section, target_section=target_section)
    return SectionComparison(
        alignment_pair=alignment_pair,
        result_type=ComparisonResultType.MODIFIED,
        similarity_score=0.75,
        details={'explanation': 'Texts differ significantly.'},
    )


@pytest.fixture
def mock_section_comparison_similar_df():
    """Returns a mock SectionComparison object representing similar sections."""
    source_section = Section(title='S_Sim', content='This is some text.', level=0, section_type=SectionType.PARAGRAPH)
    target_section = Section(title='T_Sim', content='This is some text.', level=0, section_type=SectionType.PARAGRAPH)
    alignment_pair = AlignmentPair(source_section=source_section, target_section=target_section)
    return SectionComparison(
        alignment_pair=alignment_pair,
        result_type=ComparisonResultType.SIMILAR,
        similarity_score=1.0,
        details={'explanation': 'Texts are identical.'},
    )


@pytest.fixture
def mock_section_comparison_new_df():
    """Returns a mock SectionComparison object for a new section."""
    target_section = Section(
        title='T_New', content='This is a brand new section.', level=0, section_type=SectionType.PARAGRAPH
    )
    alignment_pair = AlignmentPair(source_section=None, target_section=target_section)
    return SectionComparison(
        alignment_pair=alignment_pair,
        result_type=ComparisonResultType.NEW,
        similarity_score=0.0,
        details={'explanation': 'New section in target.'},
    )


@pytest.fixture
def mock_section_comparison_deleted_df():
    """Returns a mock SectionComparison object for a deleted section."""
    source_section = Section(
        title='S_Del', content='This section will be deleted.', level=0, section_type=SectionType.PARAGRAPH
    )
    alignment_pair = AlignmentPair(source_section=source_section, target_section=None)
    return SectionComparison(
        alignment_pair=alignment_pair,
        result_type=ComparisonResultType.DELETED,
        similarity_score=0.0,
        details={'explanation': 'Section deleted from source.'},
    )


class TestDiffFormatter:
    """Tests for the DiffFormatter class."""

    def test_formatter_initialization_default_config(self):
        formatter = DiffFormatter()
        assert formatter.config.output_format == DiffFormat.MARKDOWN
        assert formatter.config.annotation_style == AnnotationStyle.STANDARD
        assert formatter.config.include_similarity_scores is True
        assert formatter.config.show_similar_content is False

    def test_formatter_initialization_custom_config(self):
        custom_config = FormatterConfig(
            output_format=DiffFormat.HTML,
            annotation_style=AnnotationStyle.DETAILED,
            include_similarity_scores=False,
            show_similar_content=True,
        )
        formatter = DiffFormatter(config=custom_config)
        assert formatter.config.output_format == DiffFormat.HTML
        assert formatter.config.annotation_style == AnnotationStyle.DETAILED
        assert formatter.config.include_similarity_scores is False
        assert formatter.config.show_similar_content is True

    def test_format_comparisons_empty_list(self, default_config_df: FormatterConfig):
        formatter = DiffFormatter(config=default_config_df)
        comparisons: List[SectionComparison] = []
        report = formatter.format_comparisons(comparisons, title='Empty Report')
        assert 'Empty Report' in report
        assert 'Similar sections: 0' in report
        assert 'New: 0' in report

    def test_format_markdown_basic(self, mock_section_comparison_df: SectionComparison):
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_df], title='Markdown Test')
        assert '# Markdown Test' in report
        assert '## Summary' in report
        assert '- Modified sections: 1' in report
        assert '## Details' in report
        assert f'### Section 1: {mock_section_comparison_df.result_type.value.upper()}' in report
        assert f'*Similarity: {mock_section_comparison_df.similarity_score:.2f}*' in report
        assert 'Source:' in report
        assert '```' in report
        assert mock_section_comparison_df.alignment_pair.source_section.content in report
        assert 'Target:' in report
        assert mock_section_comparison_df.alignment_pair.target_section.content in report
        assert '---' in report

    def test_format_markdown_new_section(self, mock_section_comparison_new_df: SectionComparison):
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_new_df], title='New Section MD Test')
        assert f'### Section 1: {ComparisonResultType.NEW.value.upper()}' in report
        assert '**[NEW]**' in report
        assert mock_section_comparison_new_df.alignment_pair.target_section.content in report
        assert 'Source:' not in report

    def test_format_markdown_deleted_section(self, mock_section_comparison_deleted_df: SectionComparison):
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_deleted_df], title='Deleted Section MD Test')
        assert f'### Section 1: {ComparisonResultType.DELETED.value.upper()}' in report
        assert '**[DELETED]**' in report
        assert mock_section_comparison_deleted_df.alignment_pair.source_section.content in report
        assert 'Target:' not in report

    def test_format_markdown_similar_section_show_unchanged_true(
        self, mock_section_comparison_similar_df: SectionComparison
    ):
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN, show_similar_content=True)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_similar_df])
        assert f'### Section 1: {ComparisonResultType.SIMILAR.value.upper()}' in report
        assert '**[SIMILAR]**' in report
        assert mock_section_comparison_similar_df.alignment_pair.source_section.content in report

    def test_format_markdown_similar_section_show_unchanged_false(
        self, mock_section_comparison_similar_df: SectionComparison
    ):
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN, show_similar_content=False)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_similar_df])
        assert f'### Section 1: {ComparisonResultType.SIMILAR.value.upper()}' not in report
        assert '## Summary' in report
        assert '- Similar sections: 1' in report

    def test_format_markdown_detailed_annotation(self, mock_section_comparison_df: SectionComparison):
        config = FormatterConfig(output_format=DiffFormat.MARKDOWN, annotation_style=AnnotationStyle.DETAILED)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_df])
        assert 'Diff:' in report
        assert '```diff' in report
        assert '+This is the modified text.' in report or '-This is the original text.' in report

    def test_format_html_basic(self, mock_section_comparison_df: SectionComparison):
        config = FormatterConfig(output_format=DiffFormat.HTML)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_df], title='HTML Test')
        assert '<title>HTML Test</title>' in report
        assert "<div class='summary'>" in report
        assert '<li>Modified sections: 1</li>' in report
        assert '<h2>Details</h2>' in report
        assert f"<div class='section {mock_section_comparison_df.result_type.value.lower()}'>" in report
        assert f"<div class='header'>Section 1: {mock_section_comparison_df.result_type.value.upper()}</div>" in report
        assert f"<div class='similarity'>Similarity: {mock_section_comparison_df.similarity_score:.2f}</div>" in report
        assert "<div class='source'>" in report
        assert '<strong>Source:</strong>' in report
        assert self._escape_html_for_test(mock_section_comparison_df.alignment_pair.source_section.content) in report
        assert "<div class='target'>" in report
        assert '<strong>Target:</strong>' in report
        assert self._escape_html_for_test(mock_section_comparison_df.alignment_pair.target_section.content) in report

    def test_format_html_detailed_annotation(self, mock_section_comparison_df: SectionComparison):
        config = FormatterConfig(output_format=DiffFormat.HTML, annotation_style=AnnotationStyle.DETAILED)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_df])
        assert "<div class='diff'>" in report
        assert '<strong>Diff:</strong>' in report
        assert "<div class='diff-add'>" in report or "<div class='diff-remove'>" in report

    def test_format_text_basic(self, mock_section_comparison_df: SectionComparison):
        config = FormatterConfig(output_format=DiffFormat.TEXT)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_df], title='Text Test')
        assert 'TEXT TEST' in report
        assert 'SUMMARY' in report
        assert 'Modified sections: 1' in report
        assert 'DETAILS' in report
        assert f'Section 1: {mock_section_comparison_df.result_type.value.upper()}' in report
        assert f'Similarity: {mock_section_comparison_df.similarity_score:.2f}' in report
        assert 'Source:' in report
        assert mock_section_comparison_df.alignment_pair.source_section.content in report
        assert 'Target:' in report
        assert mock_section_comparison_df.alignment_pair.target_section.content in report
        assert '====' in report

    def test_format_text_detailed_annotation(self, mock_section_comparison_df: SectionComparison):
        config = FormatterConfig(output_format=DiffFormat.TEXT, annotation_style=AnnotationStyle.DETAILED)
        formatter = DiffFormatter(config=config)
        report = formatter.format_comparisons([mock_section_comparison_df])
        assert 'Diff:' in report
        assert '+This is the modified text.' in report or '-This is the original text.' in report

    def test_generate_text_diff(self, default_config_df: FormatterConfig):
        formatter = DiffFormatter(config=default_config_df)
        text1 = 'line one\nline two\nline three'
        text2 = 'line one\nline two modified\nline three'
        diff_output = formatter._generate_text_diff(text1, text2)
        assert '--- source' in diff_output
        assert '+++ target' in diff_output
        assert '-line two' in diff_output
        assert '+line two modified' in diff_output
        assert ' line one' in diff_output

    def test_generate_html_diff(self, default_config_df: FormatterConfig):
        formatter = DiffFormatter(config=default_config_df)
        text1 = 'line one\nline two'
        text2 = 'line one\nline new\nline two'
        html_diff = formatter._generate_html_diff(text1, text2)
        expected_parts = ['<div>line one</div>', "<div class='diff-add'>line new</div>", '<div>line two</div>']
        actual_parts = [part.strip() for part in html_diff.splitlines() if part.strip()]
        assert actual_parts == expected_parts
        assert "<div class='diff-add'>line new</div>" in html_diff

    def test_escape_html(self, default_config_df: FormatterConfig):
        formatter = DiffFormatter(config=default_config_df)
        input_text = '<tag>&\'"'
        expected_escaped_text = html.escape(input_text)
        assert formatter._escape_html(input_text) == expected_escaped_text

    def test_wrap_text(self):
        config = FormatterConfig(wrap_width=10)
        formatter = DiffFormatter(config=config)
        text = 'This is a long line of text.'
        wrapped = formatter._wrap_text(text)
        expected = 'This is a\nlong line\nof text.'
        assert wrapped == expected

        config_no_wrap = FormatterConfig(wrap_width=0)
        formatter_no_wrap = DiffFormatter(config=config_no_wrap)
        assert formatter_no_wrap._wrap_text(text) == text

    def _escape_html_for_test(self, text: str) -> str:
        return html.escape(text)
