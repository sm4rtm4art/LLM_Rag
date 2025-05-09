"""Module for formatting document comparison results."""

import difflib
import html
from typing import List, Optional

from llm_rag.document_processing.comparison.comparison_engine import SectionComparison
from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

from .component_protocols import IDiffFormatter
from .domain_models import AnnotationStyle, ComparisonResultType, DiffFormat, FormatterConfig

logger = get_logger(__name__)


class DiffFormatter(IDiffFormatter):
    """Formats comparison results into human-readable diffs.

    This class takes a list of `SectionComparison` objects and generates a
    human-readable report in various formats (Markdown, HTML, Text).
    It includes a summary of changes and detailed views of each section
    comparison.

    If `LLMAnalysisResult` is present in a `SectionComparison`, this
    formatter will include the LLM's categorized analysis (e.g.,
    LEGAL_EFFECT_CHANGE), its explanation, and confidence score in the
    output, providing richer insights into the nature of detected differences.
    """

    def __init__(self, config: Optional[FormatterConfig] = None):
        """Initialize the diff formatter.

        Args:
            config: Configuration for formatting behavior.
                If None, default configuration will be used.

        """
        self.config = config or FormatterConfig()
        logger.info(
            f'Initialized DiffFormatter with format={self.config.output_format.value}, '
            f'style={self.config.annotation_style.value}'
        )

    def format_comparisons(self, comparisons: List[SectionComparison], title: Optional[str] = None) -> str:
        """Format list of section comparisons into a human-readable diff.

        Depending on the configured output format, this method delegates to
        specific private methods (_format_markdown, _format_html, _format_text).
        It includes handling for an optional title and overall error management.
        The formatted output will include LLM analysis details if present in
        the `SectionComparison` objects.

        Args:
            comparisons: List of section comparisons to format.
            title: Optional title for the diff report.

        Returns:
            Formatted diff as a string.

        Raises:
            DocumentProcessingError: If formatting fails.

        """
        try:
            logger.debug(f'Formatting {len(comparisons)} section comparisons')

            if self.config.output_format == DiffFormat.MARKDOWN:
                return self._format_markdown(comparisons, title)
            elif self.config.output_format == DiffFormat.HTML:
                return self._format_html(comparisons, title)
            elif self.config.output_format == DiffFormat.TEXT:
                return self._format_text(comparisons, title)
            else:
                raise ValueError(f'Unsupported diff format: {self.config.output_format}')

        except Exception as e:
            error_msg = f'Error formatting diff: {str(e)}'
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def _format_markdown(self, comparisons: List[SectionComparison], title: Optional[str] = None) -> str:
        """Format comparisons as Markdown.

        Generates a Markdown report detailing each section comparison. If a
        section has undergone LLM analysis (i.e.,
        `comparison.llm_analysis_result` is populated), the LLM's category,
        explanation, and confidence are included in the output for that section.

        Args:
            comparisons: List of section comparisons to format.
            title: Optional title for the diff report.

        Returns:
            Markdown-formatted diff as a string.

        """
        lines = []

        # Add title
        if title:
            lines.append(f'# {title}')
            lines.append('')

        # Add summary
        similar_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.SIMILAR)
        modified_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.MODIFIED)
        different_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.DIFFERENT)
        new_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.NEW)
        deleted_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.DELETED)

        lines.append('## Summary')
        lines.append('')
        lines.append(f'- Similar sections: {similar_count}')
        lines.append(f'- Modified sections: {modified_count}')
        lines.append(f'- Different sections: {different_count}')
        lines.append(f'- New: {new_count}')
        lines.append(f'- Deleted: {deleted_count}')
        lines.append('')
        lines.append('## Details')
        lines.append('')

        # Add each section comparison
        for i, comparison in enumerate(comparisons, 1):
            # Use show_similar_content
            if not self.config.show_similar_content and comparison.result_type == ComparisonResultType.SIMILAR:
                continue

            # Section header
            lines.append(f'### Section {i}: {comparison.result_type.value.upper()}')
            lines.append('')

            # Add similarity score if configured
            if (
                self.config.include_similarity_scores
                and comparison.similarity_score is not None
                and comparison.similarity_score > 0
            ):
                lines.append(f'*Similarity: {comparison.similarity_score:.2f}*')
                lines.append('')

            # Add LLM analysis result if available
            if comparison.llm_analysis_result:
                llm_res = comparison.llm_analysis_result
                lines.append(f'**LLM Analysis:** {llm_res.comparison_category}')
                if llm_res.explanation:
                    # Ensure explanation is not overly long on one line for Markdown
                    explanation_lines = [f'  _Explanation: {line}_' for line in llm_res.explanation.splitlines()]
                    lines.extend(explanation_lines)
                if llm_res.confidence is not None:
                    lines.append(f'  _Confidence: {llm_res.confidence:.2f}_')
                lines.append('')  # Add a blank line after LLM details for better spacing

            # Format based on result type
            if comparison.result_type == ComparisonResultType.NEW:
                lines.append('**[NEW]**')
                lines.append('')
                lines.append('```')
                lines.append(comparison.alignment_pair.target_section.content)
                lines.append('```')
            elif comparison.result_type == ComparisonResultType.DELETED:
                lines.append('**[DELETED]**')
                lines.append('')
                lines.append('```')
                lines.append(comparison.alignment_pair.source_section.content)
                lines.append('```')
            elif comparison.result_type == ComparisonResultType.SIMILAR:
                lines.append('**[SIMILAR]**')
                lines.append('')
                lines.append('```')
                lines.append(comparison.alignment_pair.source_section.content)
                lines.append('```')
            else:
                # For sections with changes (MODIFIED, DIFFERENT)
                lines.append(f'**[{comparison.result_type.value.upper()}]**')
                lines.append('')

                # Source content
                lines.append('Source:')
                lines.append('```')
                lines.append(comparison.alignment_pair.source_section.content)
                lines.append('```')

                # Target content
                lines.append('Target:')
                lines.append('```')
                lines.append(comparison.alignment_pair.target_section.content)
                lines.append('```')

                # Add diff using difflib if we're using detailed annotations
                if self.config.annotation_style == AnnotationStyle.DETAILED:
                    diff = self._generate_text_diff(
                        comparison.alignment_pair.source_section.content,
                        comparison.alignment_pair.target_section.content,
                    )
                    lines.append('Diff:')
                    lines.append('```diff')
                    lines.append(diff)
                    lines.append('```')

            # Add separator
            lines.append('')
            lines.append('---')
            lines.append('')

        return '\n'.join(lines)

    def _format_html(self, comparisons: List[SectionComparison], title: Optional[str] = None) -> str:
        """Format comparisons as HTML.

        Generates an HTML report. Similar to the Markdown formatter, if LLM
        analysis results are available for a section comparison, these details
        (category, explanation, confidence) are included in the HTML output.

        Args:
            comparisons: List of section comparisons to format.
            title: Optional title for the diff report.

        Returns:
            HTML-formatted diff as a string.

        """
        # Basic HTML structure
        html = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            "<meta charset='utf-8'>",
            f'<title>{title or "Document Comparison"}</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 20px; }',
            '.section { margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }',
            '.similar { background-color: #f0f8ff; }',  # Light blue
            '.modified { background-color: #e6ffe6; }',  # Light green
            '.different { background-color: #fff0e6; }',  # Light orange
            '.new { background-color: #e6ffee; }',  # Light mint
            '.deleted { background-color: #ffe6e6; }',  # Light pink
            '.header { font-weight: bold; margin-bottom: 10px; }',
            '.content { white-space: pre-wrap; font-family: monospace; }',
            '.source { border-left: 3px solid #555; padding-left: 10px; margin-bottom: 10px; }',
            '.target { border-left: 3px solid #0077cc; padding-left: 10px; }',
            '.diff { font-family: monospace; white-space: pre-wrap; margin-top: 10px; }',
            '.diff-add { background-color: #e6ffec; color: #117700; }',
            '.diff-remove { background-color: #ffebe9; color: #cc0000; }',
            '.similarity { color: #777; font-style: italic; }',
            '.summary { margin-bottom: 20px; }',
            '</style>',
            '</head>',
            '<body>',
            f'<h1>{title or "Document Comparison"}</h1>',
        ]

        # Add summary
        similar_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.SIMILAR)
        modified_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.MODIFIED)
        different_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.DIFFERENT)
        new_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.NEW)
        deleted_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.DELETED)

        html.append("<div class='summary'>")
        html.append('<h2>Summary</h2>')
        html.append('<ul>')
        html.append(f'<li>Similar sections: {similar_count}</li>')
        html.append(f'<li>Modified sections: {modified_count}</li>')
        html.append(f'<li>Different sections: {different_count}</li>')
        html.append(f'<li>New: {new_count}</li>')
        html.append(f'<li>Deleted: {deleted_count}</li>')
        html.append('</ul>')
        html.append('</div>')

        html.append('<h2>Details</h2>')

        # Add each section comparison
        for i, comparison in enumerate(comparisons, 1):
            # Use show_similar_content
            if not self.config.show_similar_content and comparison.result_type == ComparisonResultType.SIMILAR:
                continue

            result_class = comparison.result_type.value.lower()
            html.append(f"<div class='section {result_class}'>")
            html.append(f"<div class='header'>Section {i}: {comparison.result_type.value.upper()}</div>")

            # Add similarity score if configured
            if (
                self.config.include_similarity_scores
                and comparison.similarity_score is not None
                and comparison.similarity_score > 0
            ):
                html.append(f"<div class='similarity'>Similarity: {comparison.similarity_score:.2f}</div>")

            # Add LLM analysis result if available
            if comparison.llm_analysis_result:
                llm_res = comparison.llm_analysis_result
                html.append(f"<div class='similarity'>LLM Analysis: {llm_res.comparison_category}</div>")
                if llm_res.explanation:
                    html.append(f"<div class='similarity'>Explanation: {llm_res.explanation}</div>")
                if llm_res.confidence is not None:
                    html.append(f"<div class='similarity'>Confidence: {llm_res.confidence:.2f}</div>")

            # Format based on result type
            if comparison.result_type == ComparisonResultType.NEW:
                html.append("<div class='content'>")
                html.append(self._escape_html(comparison.alignment_pair.target_section.content))
                html.append('</div>')
            elif comparison.result_type == ComparisonResultType.DELETED:
                html.append("<div class='content'>")
                html.append(self._escape_html(comparison.alignment_pair.source_section.content))
                html.append('</div>')
            elif comparison.result_type == ComparisonResultType.SIMILAR:
                html.append("<div class='content'>")
                html.append(self._escape_html(comparison.alignment_pair.source_section.content))
                html.append('</div>')
            else:
                # For sections with changes (MODIFIED, DIFFERENT)
                html.append("<div class='source'>")
                html.append('<strong>Source:</strong>')
                html.append("<div class='content'>")
                html.append(self._escape_html(comparison.alignment_pair.source_section.content))
                html.append('</div>')
                html.append('</div>')

                html.append("<div class='target'>")
                html.append('<strong>Target:</strong>')
                html.append("<div class='content'>")
                html.append(self._escape_html(comparison.alignment_pair.target_section.content))
                html.append('</div>')
                html.append('</div>')

                # Add diff if using detailed annotations
                if self.config.annotation_style == AnnotationStyle.DETAILED:
                    html_diff = self._generate_html_diff(
                        comparison.alignment_pair.source_section.content,
                        comparison.alignment_pair.target_section.content,
                    )
                    html.append("<div class='diff'>")
                    html.append('<strong>Diff:</strong>')
                    html.append(html_diff)
                    html.append('</div>')

            html.append('</div>')  # Close section div

        # Close HTML
        html.append('</body>')
        html.append('</html>')

        return '\n'.join(html)

    def _format_text(self, comparisons: List[SectionComparison], title: Optional[str] = None) -> str:
        """Format comparisons as plain text.

        Generates a plain text report. If LLM analysis results are present for
        a section comparison, its category, explanation, and confidence are
        included in the text output.

        Args:
            comparisons: List of section comparisons to format.
            title: Optional title for the diff report.

        Returns:
            Text-formatted diff as a string.

        """
        lines = []

        # Add title
        if title:
            lines.append(title.upper())
            lines.append('=' * len(title))
            lines.append('')

        # Add summary
        similar_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.SIMILAR)
        modified_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.MODIFIED)
        different_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.DIFFERENT)
        new_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.NEW)
        deleted_count = sum(1 for c in comparisons if c.result_type == ComparisonResultType.DELETED)

        lines.append('SUMMARY')
        lines.append('=======')
        lines.append('')
        lines.append(f'Similar sections: {similar_count}')
        lines.append(f'Modified sections: {modified_count}')
        lines.append(f'Different sections: {different_count}')
        lines.append(f'New: {new_count}')
        lines.append(f'Deleted: {deleted_count}')
        lines.append('')
        lines.append('DETAILS')
        lines.append('=======')
        lines.append('')

        # Add each section comparison
        for i, comparison in enumerate(comparisons, 1):
            # Use show_similar_content
            if not self.config.show_similar_content and comparison.result_type == ComparisonResultType.SIMILAR:
                continue

            # Section header
            section_title = f'Section {i}: {comparison.result_type.value.upper()}'
            lines.append(section_title)
            lines.append('-' * len(section_title))
            lines.append('')

            # Add similarity score if configured
            if self.config.include_similarity_scores and comparison.similarity_score > 0:
                lines.append(f'Similarity: {comparison.similarity_score:.2f}')
                lines.append('')

            # Add LLM analysis result if available
            if comparison.llm_analysis_result:
                llm_res = comparison.llm_analysis_result
                lines.append(f'LLM Analysis: {llm_res.comparison_category}')
                if llm_res.explanation:
                    lines.append(f'Explanation: {llm_res.explanation}')
                if llm_res.confidence is not None:
                    lines.append(f'Confidence: {llm_res.confidence:.2f}')
                lines.append('')

            # Format based on result type
            if comparison.result_type == ComparisonResultType.NEW:
                lines.append('[NEW]')
                lines.append('')
                content = self._wrap_text(comparison.alignment_pair.target_section.content)
                lines.extend(content.splitlines())
            elif comparison.result_type == ComparisonResultType.DELETED:
                lines.append('[DELETED]')
                lines.append('')
                content = self._wrap_text(comparison.alignment_pair.source_section.content)
                lines.extend(content.splitlines())
            elif comparison.result_type == ComparisonResultType.SIMILAR:
                lines.append('[SIMILAR]')
                lines.append('')
                content = self._wrap_text(comparison.alignment_pair.source_section.content)
                lines.extend(content.splitlines())
            else:
                # For sections with changes (MODIFIED, DIFFERENT)
                lines.append(f'[{comparison.result_type.value.upper()}]')
                lines.append('')

                # Source content
                lines.append('Source:')
                content = self._wrap_text(comparison.alignment_pair.source_section.content)
                lines.extend(content.splitlines())
                lines.append('')

                # Target content
                lines.append('Target:')
                content = self._wrap_text(comparison.alignment_pair.target_section.content)
                lines.extend(content.splitlines())

                # Add diff using difflib if we're using detailed annotations
                if self.config.annotation_style == AnnotationStyle.DETAILED:
                    lines.append('')
                    lines.append('Diff:')
                    diff = self._generate_text_diff(
                        comparison.alignment_pair.source_section.content,
                        comparison.alignment_pair.target_section.content,
                    )
                    lines.extend(diff.splitlines())

            # Add separator
            lines.append('')
            lines.append('=' * 80)
            lines.append('')

        return '\n'.join(lines)

    def _generate_text_diff(self, text1: str, text2: str) -> str:
        """Generate a text diff between two strings.

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Diff output as a string.

        """
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()

        diff = difflib.unified_diff(lines1, lines2, n=3, lineterm='', fromfile='source', tofile='target')

        return '\n'.join(diff)

    def _generate_html_diff(self, text1: str, text2: str) -> str:
        """Generate an HTML diff between two strings.

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            HTML diff output as a string.

        """
        # Use HtmlDiff class for side-by-side comparison
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()

        # For a more compact inline diff:
        differ = difflib.Differ()
        diff = list(differ.compare(lines1, lines2))

        html_diff = []
        for line in diff:
            if line.startswith('+ '):
                html_diff.append(f"<div class='diff-add'>{self._escape_html(line[2:])}</div>")
            elif line.startswith('- '):
                html_diff.append(f"<div class='diff-remove'>{self._escape_html(line[2:])}</div>")
            elif line.startswith('? '):
                # Skip the "?" lines which are just hints for the differ
                continue
            else:
                # Unchanged line starting with "  "
                html_diff.append(f'<div>{self._escape_html(line[2:])}</div>')

        return '\n'.join(html_diff)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Input text to escape.

        Returns:
            HTML-escaped text.

        """
        return html.escape(text)

    def _wrap_text(self, text: str) -> str:
        """Wrap text to the configured width.

        Args:
            text: Input text to wrap.

        Returns:
            Wrapped text.

        """
        if not self.config.wrap_width:
            return text

        import textwrap

        return '\n'.join(
            textwrap.fill(line, width=self.config.wrap_width, replace_whitespace=False) for line in text.splitlines()
        )
