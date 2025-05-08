"""Output formatter for OCR results.

This module provides functionality to format OCR output into more structured formats
such as Markdown, making the raw OCR text more readable and preserving basic
structural elements like paragraphs, headings, and lists.
"""

import re
from typing import Dict, List, Union

from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


class MarkdownFormatter:
    """Formats OCR text into Markdown format.

    This class provides methods to format raw OCR text into Markdown, preserving
    structural elements like paragraphs, headings, lists, and page breaks.
    """

    def __init__(self, detect_headings: bool = True, detect_lists: bool = True, detect_tables: bool = False):
        """Initialize the Markdown formatter.

        Args:
            detect_headings: Whether to attempt to detect and format headings.
            detect_lists: Whether to attempt to detect and format lists.
            detect_tables: Whether to attempt to detect and format tables (experimental).

        """
        self.detect_headings = detect_headings
        self.detect_lists = detect_lists
        self.detect_tables = detect_tables
        logger.info('Initialized Markdown formatter')

    def format_document(self, pages: Union[List[str], Dict[int, str]]) -> str:
        """Format multiple pages of OCR text into a single Markdown document.

        Args:
            pages: Either a list of text strings (one per page) or a dictionary
                mapping page numbers to text strings.

        Returns:
            Formatted Markdown text for the entire document.

        """
        # Convert dictionary to list if necessary
        if isinstance(pages, dict):
            # Sort by page number
            sorted_pages = sorted(pages.items(), key=lambda x: x[0])
            pages_list = [text for _, text in sorted_pages]
        else:
            pages_list = pages

        # Format each page and combine
        formatted_pages = []
        for i, page_text in enumerate(pages_list):
            formatted_page = self.format_page(page_text)

            # Add page header except for the first page
            if i > 0:
                formatted_pages.append(f'\n\n## Page {i + 1}\n\n{formatted_page}')
            else:
                formatted_pages.append(formatted_page)

        return '\n\n'.join(formatted_pages)

    def format_page(self, text: str) -> str:
        """Format a single page of OCR text into Markdown.

        Args:
            text: Raw OCR text for a single page.

        Returns:
            Formatted Markdown text.

        """
        if not text or not text.strip():
            return ''

        # Strip excessive whitespace
        text = text.strip()

        # Detect paragraphs (split by double newlines)
        paragraphs = self._detect_paragraphs(text)

        # Process each paragraph
        formatted_paragraphs = []
        for para in paragraphs:
            # Skip empty paragraphs
            if not para.strip():
                continue

            # Check if it's a potential heading
            if self.detect_headings and self._is_heading(para):
                formatted_para = self._format_heading(para)
            # Check if it's a potential list
            elif self.detect_lists and self._is_list_item(para):
                formatted_para = self._format_list_item(para)
            # Otherwise treat as regular paragraph
            else:
                formatted_para = para

            formatted_paragraphs.append(formatted_para)

        return '\n\n'.join(formatted_paragraphs)

    def _detect_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs based on blank lines.

        Args:
            text: Raw OCR text.

        Returns:
            List of paragraph strings.

        """
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)

        # Split by double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)

        # Remove leading/trailing whitespace from each paragraph
        paragraphs = [p.strip() for p in paragraphs]

        return paragraphs

    def _is_heading(self, text: str) -> bool:
        """Detect if a text line appears to be a heading.

        Args:
            text: Text to check.

        Returns:
            True if the text appears to be a heading, False otherwise.

        """
        # Simple heading detection rules:
        # 1. Short text (less than 100 chars)
        # 2. Doesn't end with punctuation like period, comma, semicolon
        # 3. Optional: All caps or title case

        text = text.strip()

        # Is it short?
        if len(text) > 100:
            return False

        # Does it end with sentence-ending punctuation?
        if text.endswith(('.', ',', ';', ':', '!')):
            return False

        # Is it all caps or starts with a number followed by dot?
        if text.isupper() or re.match(r'^\d+\.', text):
            return True

        # Does it match common heading patterns? (e.g., "1.1 Introduction")
        if re.match(r'^(\d+\.)+\d*\s+\w', text):
            return True

        # Check for title case (most words capitalized)
        words = text.split()
        capitalized_words = sum(1 for w in words if w and w[0].isupper())
        if len(words) > 0 and capitalized_words / len(words) > 0.7:
            return True

        return False

    def _format_heading(self, text: str) -> str:
        """Format text as a Markdown heading.

        Args:
            text: Text to format as heading.

        Returns:
            Markdown-formatted heading.

        """
        # Determine heading level based on characteristics

        # Check if it starts with numbering like "1.", "1.1.", etc.
        if re.match(r'^\d+\.', text):
            # Single number = h2
            level = 2
        elif re.match(r'^\d+\.\d+', text):
            # Two numbers (e.g., "1.1") = h3
            level = 3
        elif re.match(r'^\d+\.\d+\.\d+', text):
            # Three numbers (e.g., "1.1.1") = h4
            level = 4
        elif text.isupper():
            # ALL CAPS = h2
            level = 2
        else:
            # Default heading level
            level = 3

        # Create Markdown heading
        heading_chars = '#' * level
        return f'{heading_chars} {text}'

    def _is_list_item(self, text: str) -> bool:
        """Detect if a text line appears to be a list item.

        Args:
            text: Text to check.

        Returns:
            True if the text appears to be a list item, False otherwise.

        """
        # Check for common list markers
        lines = text.split('\n')
        first_line = lines[0].strip() if lines else ''

        # Check for bullet points or numbered lists
        return bool(
            re.match(r'^[\*\-•◦▪▫○●]', first_line)
            or re.match(r'^\d+[\.\)]', first_line)
            or re.match(r'^[a-z][\.\)]', first_line)
        )

    def _format_list_item(self, text: str) -> str:
        """Format text as a Markdown list item.

        Args:
            text: Text to format as list item.

        Returns:
            Markdown-formatted list item.

        """
        lines = text.split('\n')
        formatted_lines = []

        for i, line in enumerate(lines):
            if i == 0:
                # For the first line, preserve the list marker or convert to Markdown
                if re.match(r'^[\*\-•◦▪▫○●]', line):
                    # Convert various bullet styles to Markdown bullet
                    line = re.sub(r'^[\*\-•◦▪▫○●]\s*', '* ', line)
                elif re.match(r'^\d+[\.\)]', line):
                    # Convert numbered list to Markdown numbered list
                    line = re.sub(r'^\d+[\.\)]\s*', '1. ', line)
                elif re.match(r'^[a-z][\.\)]', line):
                    # Convert lettered list to bullet list
                    line = re.sub(r'^[a-z][\.\)]\s*', '* ', line)
            else:
                # For continuation lines, add proper indentation
                line = '  ' + line

            formatted_lines.append(line)

        return '\n'.join(formatted_lines)


class JSONFormatter:
    """Formats OCR text into structured JSON format.

    This is a placeholder for future implementation of JSON formatting
    which would be part of Phase 5 according to the expansion plan.
    """

    def __init__(self):
        """Initialize the JSON formatter."""
        logger.info('Initialized JSON formatter (placeholder)')

    def format_document(self, pages: Union[List[str], Dict[int, str]]) -> Dict:
        """Format OCR text into a structured JSON document.

        Args:
            pages: Either a list of text strings (one per page) or a dictionary
                mapping page numbers to text strings.

        Returns:
            Structured JSON representation of the document.

        """
        # Basic implementation for now
        if isinstance(pages, dict):
            result = {'pages': {}}
            for page_num, text in pages.items():
                result['pages'][str(page_num)] = {'text': text}
        else:
            result = {'pages': {}}
            for i, text in enumerate(pages):
                result['pages'][str(i)] = {'text': text}

        return result
