"""Module for parsing structured documents into logical sections."""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


class SectionType(Enum):
    """Types of document sections."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    IMAGE = "image"
    UNKNOWN = "unknown"


@dataclass
class Section:
    """Representation of a document section.

    Attributes:
        id: Unique identifier for the section.
        type: Type of the section (heading, paragraph, etc.).
        content: The text content of the section.
        level: Section level (e.g., heading level).
        metadata: Additional information about the section.
        parent_id: ID of the parent section, if any.
        children: IDs of child sections, if any.

    """

    id: str
    type: SectionType
    content: str
    level: Optional[int] = None
    metadata: Optional[Dict] = None
    parent_id: Optional[str] = None
    children: Optional[List[str]] = None


@dataclass
class ParserConfig:
    """Configuration for document parsing.

    Attributes:
        min_section_length: Minimum content length to be considered a valid section.
        max_section_length: Maximum content length for a section before splitting.
        heading_patterns: Regular expressions for identifying headings.
        split_on_headings: Whether to split the document at headings.
        split_on_blank_lines: Whether to split paragraphs at blank lines.
        keep_whitespace: Whether to preserve whitespace in the parsed sections.
        language: Document language for specialized processing.

    """

    min_section_length: int = 10
    max_section_length: int = 2000
    heading_patterns: List[str] = None
    split_on_headings: bool = True
    split_on_blank_lines: bool = True
    keep_whitespace: bool = False
    language: str = "en"

    def __post_init__(self):
        """Initialize default values for heading patterns if not provided."""
        if self.heading_patterns is None:
            # Default Markdown heading patterns
            self.heading_patterns = [
                r"^#{1,6}\s+(.+)$",  # ATX style headings: # Heading
                r"^(.+)\n[=]{2,}$",  # Setext style heading level 1: Heading\n======
                r"^(.+)\n[-]{2,}$",  # Setext style heading level 2: Heading\n------
            ]


class DocumentFormat(Enum):
    """Supported document formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


class DocumentParser:
    """Parser for structured documents into logical sections."""

    def __init__(self, config: Optional[ParserConfig] = None):
        """Initialize the document parser.

        Args:
            config: Configuration for parsing behavior.
                If None, default configuration will be used.

        """
        self.config = config or ParserConfig()
        self._heading_patterns = [re.compile(p, re.MULTILINE) for p in self.config.heading_patterns]
        logger.info(
            f"Initialized DocumentParser with split_on_headings={self.config.split_on_headings}, "
            f"min_section_length={self.config.min_section_length}"
        )

    def parse(self, document: str, format_type: DocumentFormat = DocumentFormat.MARKDOWN) -> List[Section]:
        """Parse document into sections.

        Args:
            document: The document content as a string.
            format_type: Format of the document (markdown, json, etc.).

        Returns:
            List of document sections.

        Raises:
            DocumentProcessingError: If parsing fails.

        """
        try:
            logger.debug(f"Parsing document with format: {format_type.value}")

            if format_type == DocumentFormat.MARKDOWN:
                sections = self._parse_markdown(document)
            elif format_type == DocumentFormat.JSON:
                sections = self._parse_json(document)
            elif format_type == DocumentFormat.TEXT:
                sections = self._parse_text(document)
            else:
                raise ValueError(f"Unsupported document format: {format_type}")

            logger.debug(f"Parsed {len(sections)} sections from document")
            return sections

        except Exception as e:
            error_msg = f"Error parsing document: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def _parse_markdown(self, markdown_content: str) -> List[Section]:
        """Parse Markdown document into sections.

        Args:
            markdown_content: The Markdown document as a string.

        Returns:
            List of document sections.

        """
        sections = []
        current_id = 0
        lines = markdown_content.splitlines()

        i = 0
        while i < len(lines):
            # Check for headings
            heading_match, heading_level = self._match_heading(lines[i])

            if heading_match and self.config.split_on_headings:
                # Found a heading, create a new section
                heading_text = heading_match.group(1).strip()
                current_id += 1
                section_id = f"s{current_id}"

                sections.append(
                    Section(
                        id=section_id,
                        type=SectionType.HEADING,
                        content=heading_text,
                        level=heading_level,
                    )
                )

                # Collect content until the next heading or end of document
                content_lines = []
                i += 1
                while i < len(lines):
                    next_heading, _ = self._match_heading(lines[i])
                    if next_heading and self.config.split_on_headings:
                        break
                    content_lines.append(lines[i])
                    i += 1

                # Create a section for the content if not empty
                content = "\n".join(content_lines).strip()
                if content and len(content) >= self.config.min_section_length:
                    current_id += 1
                    content_section_id = f"s{current_id}"
                    sections.append(
                        Section(
                            id=content_section_id,
                            type=SectionType.PARAGRAPH,
                            content=content,
                            parent_id=section_id,
                        )
                    )
                    # Update the heading section with this child
                    for section in sections:
                        if section.id == section_id:
                            if section.children is None:
                                section.children = []
                            section.children.append(content_section_id)
                            break
            else:
                # Not a heading or not splitting on headings
                # Collect content until blank line or next heading
                content_lines = [lines[i]]
                i += 1
                while i < len(lines):
                    next_heading, _ = self._match_heading(lines[i])
                    if next_heading and self.config.split_on_headings:
                        break
                    if self.config.split_on_blank_lines and not lines[i].strip():
                        i += 1  # Skip the blank line
                        break
                    content_lines.append(lines[i])
                    i += 1

                # Create a section for the content if not empty
                content = "\n".join(content_lines).strip()
                if content and len(content) >= self.config.min_section_length:
                    current_id += 1
                    sections.append(
                        Section(
                            id=f"s{current_id}",
                            type=SectionType.PARAGRAPH,
                            content=content,
                        )
                    )

        return sections

    def _match_heading(self, line: str) -> Tuple[Optional[re.Match], Optional[int]]:
        """Match a line against heading patterns.

        Args:
            line: The line to check for heading patterns.

        Returns:
            Tuple of (match_object, heading_level) or (None, None) if no match.

        """
        # Check ATX style headings (# Heading)
        atx_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if atx_match:
            level = len(atx_match.group(1))
            # Recreate match with the full heading text as group 1
            return re.match(r"^#{1,6}\s+(.+)$", line), level

        # Setext style headings need the next line, can't check in isolation
        # This is a limitation of the current implementation

        return None, None

    def _parse_json(self, json_content: str) -> List[Section]:
        """Parse JSON document into sections.

        Args:
            json_content: The JSON document as a string.

        Returns:
            List of document sections.

        """
        # In a real implementation, this would parse the JSON structure
        # and convert it to sections. For the MVP, we'll raise an error.
        raise NotImplementedError("JSON parsing is not implemented in this version")

    def _parse_text(self, text_content: str) -> List[Section]:
        """Parse plain text document into sections.

        Args:
            text_content: The text document as a string.

        Returns:
            List of document sections.

        """
        sections = []
        current_id = 0

        # Simple paragraph splitting on blank lines
        paragraphs = re.split(r"\n\s*\n", text_content)

        for paragraph in paragraphs:
            content = paragraph.strip()
            if content and len(content) >= self.config.min_section_length:
                current_id += 1
                sections.append(
                    Section(
                        id=f"s{current_id}",
                        type=SectionType.PARAGRAPH,
                        content=content,
                    )
                )

        return sections

    def _load_document(self, document_path: Union[str, Path]) -> str:
        """Load a document from a file path.

        Args:
            document_path: Path to the document file.

        Returns:
            The content of the document as a string.

        Raises:
            DocumentProcessingError: If the document cannot be loaded.

        """
        try:
            path = Path(document_path)

            # Check if file exists
            if not path.exists():
                error_msg = f"Document file not found: {document_path}"
                logger.error(error_msg)
                raise DocumentProcessingError(error_msg)

            # Check if it's a file
            if not path.is_file():
                error_msg = f"Document path is not a file: {document_path}"
                logger.error(error_msg)
                raise DocumentProcessingError(error_msg)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            logger.debug(f"Loaded document with {len(content)} characters")
            return content

        except DocumentProcessingError:
            # Re-raise existing DocumentProcessingError
            raise
        except Exception as e:
            error_msg = f"Error loading document from {document_path}: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def segment_by_fixed_chunks(self, content: str, chunk_size: int = 1000) -> List[Section]:
        """Segment text into fixed-size chunks.

        Args:
            content: Text content as a string.
            chunk_size: Maximum size of each chunk in characters.

        Returns:
            A list of Section objects representing the chunks.

        """
        logger.debug(f"Segmenting text into fixed chunks of size {chunk_size}")
        sections = []

        # Ensure we're working with a string
        if not isinstance(content, str):
            content = str(content)

        # If content is smaller than chunk_size, just return it as one chunk
        if len(content) <= chunk_size:
            return [Section(content.strip(), SectionType.PARAGRAPH)]

        # Split content into paragraphs first to maintain some coherence
        paragraphs = content.split("\n\n")
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size and we already have content
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > chunk_size:
                # Add the current chunk and start a new one
                sections.append(Section(current_chunk.strip(), SectionType.PARAGRAPH))
                current_chunk = paragraph + "\n\n"
            elif len(paragraph) > chunk_size:
                # If the paragraph itself is too long, split it by sentences or just characters
                if current_chunk:
                    sections.append(Section(current_chunk.strip(), SectionType.PARAGRAPH))
                    current_chunk = ""

                # Try to split by sentences first
                sentences = paragraph.replace(". ", ".\n").split("\n")
                current_sentence_chunk = ""

                for sentence in sentences:
                    if len(current_sentence_chunk) + len(sentence) + 1 > chunk_size:
                        if current_sentence_chunk:
                            sections.append(Section(current_sentence_chunk.strip(), SectionType.PARAGRAPH))
                            current_sentence_chunk = sentence + " "
                        else:
                            # Even a single sentence is too long, so split by characters
                            for i in range(0, len(sentence), chunk_size):
                                chunk = sentence[i : i + chunk_size]
                                sections.append(Section(chunk.strip(), SectionType.PARAGRAPH))
                    else:
                        current_sentence_chunk += sentence + " "

                if current_sentence_chunk:
                    sections.append(Section(current_sentence_chunk.strip(), SectionType.PARAGRAPH))
            else:
                # This paragraph fits, add it to the current chunk
                current_chunk += paragraph + "\n\n"

        # Add the last chunk if it's not empty
        if current_chunk.strip():
            sections.append(Section(current_chunk.strip(), SectionType.PARAGRAPH))

        logger.debug(f"Created {len(sections)} fixed-size chunks")
        return sections
