"""Module for formatting document comparison results."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from llm_rag.document_processing.comparison.comparison_engine import ComparisonResult, SectionComparison
from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


class FormatStyle(Enum):
    """Styles for formatting comparison results."""

    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN = "plain"


@dataclass
class FormatterConfig:
    """Configuration for the diff formatter.

    Attributes:
        style: Output format style.
        show_similarity_scores: Whether to include similarity scores in output.
        context_lines: Number of context lines to include in diffs.
        color_output: Whether to include color in the output (for supported formats).
        detail_level: Level of detail in the output (1=minimal, 3=detailed).

    """

    style: FormatStyle = FormatStyle.MARKDOWN
    show_similarity_scores: bool = True
    context_lines: int = 2
    color_output: bool = True
    detail_level: int = 2


class DiffFormatter:
    """Formats document comparison results into human-readable formats.

    This class takes the output of the comparison engine and produces
    formatted output in various styles (Markdown, HTML, etc.).
    """

    def __init__(self, config: Optional[FormatterConfig] = None):
        """Initialize the diff formatter.

        Args:
            config: Configuration for formatting behavior.
                If None, default configuration will be used.

        """
        self.config = config or FormatterConfig()
        logger.info(f"Initialized DiffFormatter with style: {self.config.style.value}")

    def format_comparisons(self, comparisons: List[SectionComparison], title: Optional[str] = None) -> str:
        """Format a list of section comparisons into a human-readable output.

        Args:
            comparisons: List of section comparisons to format.
            title: Optional title for the diff output.

        Returns:
            Formatted comparison output as a string.

        Raises:
            DocumentProcessingError: If formatting fails.

        """
        try:
            logger.debug(f"Formatting {len(comparisons)} section comparisons")

            # Choose formatter based on style
            if self.config.style == FormatStyle.MARKDOWN:
                return self._format_markdown(comparisons, title)
            elif self.config.style == FormatStyle.HTML:
                return self._format_html(comparisons, title)
            elif self.config.style == FormatStyle.PLAIN:
                return self._format_plain(comparisons, title)
            else:
                raise ValueError(f"Unsupported format style: {self.config.style}")

        except Exception as e:
            error_msg = f"Error formatting comparison results: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def _format_markdown(self, comparisons: List[SectionComparison], title: Optional[str]) -> str:
        """Format comparisons as Markdown.

        Args:
            comparisons: List of section comparisons to format.
            title: Optional title for the output.

        Returns:
            Formatted Markdown string.

        """
        logger.debug("Formatting comparisons as Markdown")

        lines = []

        # Add title if provided
        if title:
            lines.append(f"# {title}")
            lines.append("")

        # Add summary
        lines.append("## Summary")
        lines.append("")
        lines.extend(self._generate_summary(comparisons))
        lines.append("")

        # Add detailed comparisons
        lines.append("## Detailed Comparison")
        lines.append("")

        for i, comparison in enumerate(comparisons):
            section_header = f"### Section {i + 1}"

            if (
                comparison.alignment_pair.source_section
                and comparison.alignment_pair.source_section.section_type.value == "heading"
            ):
                # Use the heading text for the section header
                heading_text = comparison.alignment_pair.source_section.content
                section_header = f"### {heading_text}"
            elif (
                comparison.alignment_pair.target_section
                and comparison.alignment_pair.target_section.section_type.value == "heading"
            ):
                heading_text = comparison.alignment_pair.target_section.content
                section_header = f"### {heading_text}"

            lines.append(section_header)

            # Add result classification
            result = comparison.result.value.upper()
            if self.config.show_similarity_scores and comparison.similarity_score > 0:
                result_line = f"**Result**: {result} (Similarity: {comparison.similarity_score:.2f})"
            else:
                result_line = f"**Result**: {result}"
            lines.append(result_line)
            lines.append("")

            # Format the section content based on the result
            if comparison.result == ComparisonResult.SIMILAR:
                if self.config.detail_level >= 2:
                    lines.append("Content is similar in both documents:")
                    lines.append("```")
                    lines.append(comparison.alignment_pair.source_section.content)
                    lines.append("```")
            elif comparison.result == ComparisonResult.DELETED:
                lines.append("Content present only in source document:")
                lines.append("```diff")
                lines.append("- " + comparison.alignment_pair.source_section.content.replace("\n", "\n- "))
                lines.append("```")
            elif comparison.result == ComparisonResult.NEW:
                lines.append("Content present only in target document:")
                lines.append("```diff")
                lines.append("+ " + comparison.alignment_pair.target_section.content.replace("\n", "\n+ "))
                lines.append("```")
            else:
                # Modified content - show diff
                lines.append("Content differs between documents:")
                lines.append("```diff")
                lines.append("- " + comparison.alignment_pair.source_section.content.replace("\n", "\n- "))
                lines.append("+ " + comparison.alignment_pair.target_section.content.replace("\n", "\n+ "))
                lines.append("```")

            lines.append("")

        return "\n".join(lines)

    def _format_html(self, comparisons: List[SectionComparison], title: Optional[str]) -> str:
        """Format comparisons as HTML.

        Args:
            comparisons: List of section comparisons to format.
            title: Optional title for the output.

        Returns:
            Formatted HTML string.

        """
        logger.debug("Formatting comparisons as HTML")

        lines = []

        # HTML header
        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append('  <meta charset="UTF-8">')
        if title:
            lines.append(f"  <title>{title}</title>")
        else:
            lines.append("  <title>Document Comparison</title>")

        # Add some basic styling
        lines.append("  <style>")
        lines.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
        lines.append("    .similar { background-color: #e6ffe6; }")
        lines.append("    .minor-changes { background-color: #ffffcc; }")
        lines.append("    .major-changes { background-color: #ffe6cc; }")
        lines.append("    .rewritten { background-color: #ffe6e6; }")
        lines.append("    .new { background-color: #e6fff2; }")
        lines.append("    .deleted { background-color: #f2e6ff; }")
        lines.append("    .diff-source { color: #cc0000; text-decoration: line-through; }")
        lines.append("    .diff-target { color: #00cc00; }")
        lines.append("    .section { border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; }")
        lines.append("    .result { font-weight: bold; margin-bottom: 10px; }")
        lines.append("    .summary { margin-bottom: 30px; }")
        lines.append("    pre { white-space: pre-wrap; }")
        lines.append("  </style>")
        lines.append("</head>")
        lines.append("<body>")

        # Add title
        if title:
            lines.append(f"<h1>{title}</h1>")

        # Add summary
        lines.append('<div class="summary">')
        lines.append("<h2>Summary</h2>")
        summary_lines = self._generate_summary(comparisons)
        lines.append("<ul>")
        for line in summary_lines:
            lines.append(f"  <li>{line}</li>")
        lines.append("</ul>")
        lines.append("</div>")

        # Add detailed comparisons
        lines.append("<h2>Detailed Comparison</h2>")

        for i, comparison in enumerate(comparisons):
            result_class = comparison.result.value.replace("_", "-")
            lines.append(f'<div class="section {result_class}">')

            # Section header
            if (
                comparison.alignment_pair.source_section
                and comparison.alignment_pair.source_section.section_type.value == "heading"
            ):
                heading_text = comparison.alignment_pair.source_section.content
                lines.append(f"<h3>{heading_text}</h3>")
            elif (
                comparison.alignment_pair.target_section
                and comparison.alignment_pair.target_section.section_type.value == "heading"
            ):
                heading_text = comparison.alignment_pair.target_section.content
                lines.append(f"<h3>{heading_text}</h3>")
            else:
                lines.append(f"<h3>Section {i + 1}</h3>")

            # Result classification
            result = comparison.result.value.upper().replace("_", " ")
            if self.config.show_similarity_scores and comparison.similarity_score > 0:
                lines.append(
                    f'<div class="result">Result: {result} (Similarity: {comparison.similarity_score:.2f})</div>'
                )
            else:
                lines.append(f'<div class="result">Result: {result}</div>')

            # Format content based on result
            if comparison.result == ComparisonResult.SIMILAR:
                if self.config.detail_level >= 2:
                    lines.append("<p>Content is similar in both documents:</p>")
                    lines.append("<pre>")
                    lines.append(self._escape_html(comparison.alignment_pair.source_section.content))
                    lines.append("</pre>")
            elif comparison.result == ComparisonResult.DELETED:
                lines.append("<p>Content present only in source document:</p>")
                lines.append('<pre class="diff-source">')
                lines.append(self._escape_html(comparison.alignment_pair.source_section.content))
                lines.append("</pre>")
            elif comparison.result == ComparisonResult.NEW:
                lines.append("<p>Content present only in target document:</p>")
                lines.append('<pre class="diff-target">')
                lines.append(self._escape_html(comparison.alignment_pair.target_section.content))
                lines.append("</pre>")
            else:
                # Modified content - show both versions
                lines.append("<p>Content differs between documents:</p>")
                lines.append("<div>")
                lines.append('<pre class="diff-source">')
                lines.append(self._escape_html(comparison.alignment_pair.source_section.content))
                lines.append("</pre>")
                lines.append('<pre class="diff-target">')
                lines.append(self._escape_html(comparison.alignment_pair.target_section.content))
                lines.append("</pre>")
                lines.append("</div>")

            lines.append("</div>")

        # HTML footer
        lines.append("</body>")
        lines.append("</html>")

        return "\n".join(lines)

    def _format_plain(self, comparisons: List[SectionComparison], title: Optional[str]) -> str:
        """Format comparisons as plain text.

        Args:
            comparisons: List of section comparisons to format.
            title: Optional title for the output.

        Returns:
            Formatted plain text string.

        """
        logger.debug("Formatting comparisons as plain text")

        lines = []

        # Add title if provided
        if title:
            lines.append(title)
            lines.append("=" * len(title))
            lines.append("")

        # Add summary
        lines.append("SUMMARY")
        lines.append("=======")
        lines.append("")
        lines.extend(self._generate_summary(comparisons))
        lines.append("")

        # Add detailed comparisons
        lines.append("DETAILED COMPARISON")
        lines.append("===================")
        lines.append("")

        for i, comparison in enumerate(comparisons):
            section_header = f"Section {i + 1}"

            if (
                comparison.alignment_pair.source_section
                and comparison.alignment_pair.source_section.section_type.value == "heading"
            ):
                heading_text = comparison.alignment_pair.source_section.content
                section_header = heading_text
            elif (
                comparison.alignment_pair.target_section
                and comparison.alignment_pair.target_section.section_type.value == "heading"
            ):
                heading_text = comparison.alignment_pair.target_section.content
                section_header = heading_text

            lines.append(section_header)
            lines.append("-" * len(section_header))

            # Add result classification
            result = comparison.result.value.upper()
            if self.config.show_similarity_scores and comparison.similarity_score > 0:
                result_line = f"Result: {result} (Similarity: {comparison.similarity_score:.2f})"
            else:
                result_line = f"Result: {result}"
            lines.append(result_line)
            lines.append("")

            # Format the section content based on the result
            if comparison.result == ComparisonResult.SIMILAR:
                if self.config.detail_level >= 2:
                    lines.append("Content is similar in both documents:")
                    lines.append(comparison.alignment_pair.source_section.content)
            elif comparison.result == ComparisonResult.DELETED:
                lines.append("Content present only in source document:")
                lines.append("- " + comparison.alignment_pair.source_section.content.replace("\n", "\n- "))
            elif comparison.result == ComparisonResult.NEW:
                lines.append("Content present only in target document:")
                lines.append("+ " + comparison.alignment_pair.target_section.content.replace("\n", "\n+ "))
            else:
                # Modified content - show diff
                lines.append("Content differs between documents:")
                lines.append("SOURCE:")
                lines.append("- " + comparison.alignment_pair.source_section.content.replace("\n", "\n- "))
                lines.append("")
                lines.append("TARGET:")
                lines.append("+ " + comparison.alignment_pair.target_section.content.replace("\n", "\n+ "))

            lines.append("")
            lines.append("")

        return "\n".join(lines)

    def _generate_summary(self, comparisons: List[SectionComparison]) -> List[str]:
        """Generate a summary of the comparison results.

        Args:
            comparisons: List of section comparisons.

        Returns:
            List of summary text lines.

        """
        # Count by result type
        counts = {result_type: 0 for result_type in ComparisonResult}
        for comparison in comparisons:
            counts[comparison.result] += 1

        summary_lines = []
        summary_lines.append(f"Total sections: {len(comparisons)}")
        summary_lines.append(f"Similar sections: {counts[ComparisonResult.SIMILAR]}")
        summary_lines.append(f"Sections with minor changes: {counts[ComparisonResult.MINOR_CHANGES]}")
        summary_lines.append(f"Sections with major changes: {counts[ComparisonResult.MAJOR_CHANGES]}")
        summary_lines.append(f"Rewritten sections: {counts[ComparisonResult.REWRITTEN]}")
        summary_lines.append(f"New sections: {counts[ComparisonResult.NEW]}")
        summary_lines.append(f"Deleted sections: {counts[ComparisonResult.DELETED]}")

        return summary_lines

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters in a string.

        Args:
            text: The text to escape.

        Returns:
            HTML-escaped text.

        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
