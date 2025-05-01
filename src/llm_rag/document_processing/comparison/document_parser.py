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


class Section:
    """Representation of a document section.

    Attributes:
        id: Unique identifier for the section.
        section_type: Type of the section (heading, paragraph, etc.).
        content: The text content of the section.
        level: Section level (e.g., heading level).
        metadata: Additional information about the section.
        parent_id: ID of the parent section, if any.
        children: IDs of child sections, if any.

    """

    def __init__(
        self,
        id: str,
        content: str,
        section_type: Optional[SectionType] = None,
        type: Optional[SectionType] = None,  # For backward compatibility
        level: Optional[int] = None,
        metadata: Optional[Dict] = None,
        parent_id: Optional[str] = None,
        children: Optional[List[str]] = None,
    ):
        """Initialize a Section.

        Args:
            id: Unique identifier for the section.
            content: The text content of the section.
            section_type: Type of the section (heading, paragraph, etc.).
            type: For backward compatibility, alias for section_type.
            level: Section level (e.g., heading level).
            metadata: Additional information about the section.
            parent_id: ID of the parent section, if any.
            children: IDs of child sections, if any.

        """
        self.id = id

        # Handle backward compatibility for type vs section_type
        if section_type is not None:
            self.section_type = section_type
        elif type is not None:
            self.section_type = type
        else:
            self.section_type = SectionType.UNKNOWN

        self.content = content
        self.level = level
        self.metadata = metadata
        self.parent_id = parent_id
        self.children = children

    def __eq__(self, other):
        """Check if two sections are equal."""
        if not isinstance(other, Section):
            return False
        return (
            self.id == other.id
            and self.section_type == other.section_type
            and self.content == other.content
            and self.level == other.level
            and self.metadata == other.metadata
            and self.parent_id == other.parent_id
            and self.children == other.children
        )

    def __repr__(self):
        """Return string representation of the Section."""
        return (
            f"Section(id='{self.id}', "
            f"section_type={self.section_type}, "
            f"content='{self.content[:30]}...', "
            f"level={self.level})"
        )


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

    def __init__(self, config: Optional[ParserConfig] = None, default_format: DocumentFormat = DocumentFormat.MARKDOWN):
        """Initialize the document parser.

        Args:
            config: Configuration for parsing behavior.
                If None, default configuration will be used.
            default_format: Default format to use when parsing documents.

        """
        self.config = config or ParserConfig()
        self.default_format = default_format
        self._heading_patterns = [re.compile(p, re.MULTILINE) for p in self.config.heading_patterns]
        logger.info(
            f"Initialized DocumentParser with split_on_headings={self.config.split_on_headings}, "
            f"min_section_length={self.config.min_section_length}"
        )

    def parse(self, document: Union[str, Path], format: Optional[DocumentFormat] = None) -> List[Section]:
        """Parse document into sections.

        Args:
            document: The document content as a string or a file path.
            format: Format of the document (markdown, json, etc.).
                If None, will try to guess the format or use default.

        Returns:
            List of document sections.

        Raises:
            DocumentProcessingError: If parsing fails.

        """
        try:
            # If format not specified, use default
            format = format or self.default_format
            logger.debug(f"Parsing document with format: {format.value}")

            # Check if document is a file path
            if isinstance(document, Path) or (
                isinstance(document, str)
                and ("/" in document or "\\" in document)
                and not document.startswith("{")
                and not document.startswith("#")
            ):
                try:
                    path = Path(document)
                    if path.exists() and path.is_file():
                        document = self._load_document(path)
                except (ValueError, OSError, PermissionError) as e:
                    # Catch specific exceptions when dealing with paths
                    logger.debug(f"Treating as content, not a valid file path: {e}")
                    # Continue with document as content

            if format == DocumentFormat.MARKDOWN:
                sections = self._parse_markdown(document)
            elif format == DocumentFormat.JSON:
                sections = self._parse_json(document)
            elif format == DocumentFormat.TEXT:
                sections = self._parse_text(document)
            else:
                raise ValueError(f"Unsupported document format: {format}")

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

        # Debug the content
        logger.debug(f"Parsing markdown with {len(lines)} lines")
        for line_num, line in enumerate(lines):
            logger.debug(f"Line {line_num}: {repr(line)}")

        i = 0
        in_code_block = False
        code_start = -1

        while i < len(lines):
            # Debug current line
            logger.debug(f"Processing line {i}: {repr(lines[i])}")

            # Handle code blocks
            if lines[i].strip() == "```" or lines[i].strip().startswith("```"):
                if not in_code_block:
                    # Start of code block
                    in_code_block = True
                    code_start = i
                    i += 1
                    logger.debug(f"Code block start at line {code_start}")
                    continue
                else:
                    # End of code block
                    in_code_block = False

                    # Extract language if specified
                    language = ""
                    if lines[code_start].strip() != "```":
                        language = lines[code_start].strip().replace("```", "").strip()

                    # Extract code content
                    code_lines = lines[code_start + 1 : i]

                    # Create code section
                    current_id += 1
                    code_content = "\n".join(code_lines)
                    sections.append(
                        Section(
                            id=f"s{current_id}",
                            section_type=SectionType.CODE,
                            content=code_content,
                            metadata={"language": language} if language else None,
                        )
                    )
                    logger.debug(f"Added code block: {code_content[:30]}...")
                    i += 1
                    continue

            # If we're in a code block, continue to next line
            if in_code_block:
                i += 1
                continue

            # Check for headings
            heading_match, heading_level = self._match_heading(lines[i])

            # Check for lists (- or * or 1. style)
            is_list_item = bool(re.match(r"^\s*[-*+]\s+.+$", lines[i]) or re.match(r"^\s*\d+\.\s+.+$", lines[i]))

            if is_list_item:
                # Found a list item, collect all items in this list
                list_items = [lines[i].strip()]
                i += 1

                # Collect subsequent list items
                while i < len(lines) and (
                    re.match(r"^\s*[-*+]\s+.+$", lines[i])
                    or re.match(r"^\s*\d+\.\s+.+$", lines[i])
                    or not lines[i].strip()  # Empty lines within list
                ):
                    if lines[i].strip():  # Skip empty lines in the count
                        list_items.append(lines[i].strip())
                    i += 1

                # Create a section for the list
                current_id += 1
                list_content = "\n".join(list_items)
                if list_content and len(list_content) >= self.config.min_section_length:
                    sections.append(
                        Section(
                            id=f"s{current_id}",
                            section_type=SectionType.LIST,
                            content=list_content,
                        )
                    )
                    logger.debug(f"Added list: {list_content[:30]}...")
                # i is already incremented in the loop
                continue

            if heading_match and self.config.split_on_headings:
                # Found a heading, create a new section
                heading_text = heading_match.group(1).strip()
                current_id += 1
                section_id = f"s{current_id}"

                sections.append(
                    Section(
                        id=section_id,
                        section_type=SectionType.HEADING,
                        content=heading_text,
                        level=heading_level,
                    )
                )
                logger.debug(f"Added heading: {heading_text}")

                # Collect content until the next heading or end of document
                content_lines = []
                i += 1
                while i < len(lines):
                    next_heading, _ = self._match_heading(lines[i])
                    is_next_list = bool(
                        re.match(r"^\s*[-*+]\s+.+$", lines[i]) or re.match(r"^\s*\d+\.\s+.+$", lines[i])
                    )
                    is_code_block = lines[i].strip() == "```" or lines[i].strip().startswith("```")

                    if (next_heading and self.config.split_on_headings) or is_next_list or is_code_block:
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
                            section_type=SectionType.PARAGRAPH,
                            content=content,
                            parent_id=section_id,
                        )
                    )
                    logger.debug(f"Added paragraph: {content[:30]}...")
                    # Update the heading section with this child
                    for section in sections:
                        if section.id == section_id:
                            if section.children is None:
                                section.children = []
                            section.children.append(content_section_id)
                            break
            else:
                # Not a heading, list, or code block
                # Collect content until blank line or next heading or list
                content_lines = [lines[i]]
                i += 1
                while i < len(lines):
                    next_heading, _ = self._match_heading(lines[i])
                    is_next_list = bool(
                        re.match(r"^\s*[-*+]\s+.+$", lines[i]) or re.match(r"^\s*\d+\.\s+.+$", lines[i])
                    )
                    is_code_block = lines[i].strip() == "```" or lines[i].strip().startswith("```")

                    if (next_heading and self.config.split_on_headings) or is_next_list or is_code_block:
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
                            section_type=SectionType.PARAGRAPH,
                            content=content,
                        )
                    )
                    logger.debug(f"Added paragraph: {content[:30]}...")

        # Check for unclosed code block
        if in_code_block:
            # Extract code content
            code_lines = lines[code_start + 1 :]

            # Create code section
            current_id += 1
            code_content = "\n".join(code_lines)
            sections.append(
                Section(id=f"s{current_id}", section_type=SectionType.CODE, content=code_content, metadata=None)
            )
            logger.debug(f"Added unclosed code block: {code_content[:30]}...")

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
        import json

        sections = []
        current_id = 0

        try:
            data = json.loads(json_content)

            # If there's a title, add it as a heading
            if "title" in data:
                current_id += 1
                sections.append(
                    Section(id=f"s{current_id}", section_type=SectionType.HEADING, content=data["title"], level=1)
                )

            # Process sections if available
            if "sections" in data and isinstance(data["sections"], list):
                for section_data in data["sections"]:
                    if "heading" in section_data:
                        current_id += 1
                        heading_id = f"s{current_id}"
                        sections.append(
                            Section(
                                id=heading_id,
                                section_type=SectionType.HEADING,
                                content=section_data["heading"],
                                level=2,
                            )
                        )

                    if "content" in section_data:
                        current_id += 1
                        content_id = f"s{current_id}"
                        parent_id = heading_id if "heading" in section_data else None
                        sections.append(
                            Section(
                                id=content_id,
                                section_type=SectionType.PARAGRAPH,
                                content=section_data["content"],
                                parent_id=parent_id,
                            )
                        )

            # If no sections were added, create a section with the entire JSON
            if not sections:
                current_id += 1
                sections.append(
                    Section(id=f"s{current_id}", section_type=SectionType.PARAGRAPH, content=json.dumps(data, indent=2))
                )

            return sections

        except json.JSONDecodeError:
            # If not valid JSON, treat as text
            current_id += 1
            return [Section(id=f"s{current_id}", section_type=SectionType.PARAGRAPH, content=json_content)]

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
                        section_type=SectionType.PARAGRAPH,
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
            return [Section(id="s1", section_type=SectionType.PARAGRAPH, content=content.strip())]

        # Split content into paragraphs first to maintain some coherence
        paragraphs = content.split("\n\n")
        current_chunk = ""
        section_id = 0

        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size and we already have content
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > chunk_size:
                # Add the current chunk and start a new one
                section_id += 1
                sections.append(
                    Section(id=f"s{section_id}", section_type=SectionType.PARAGRAPH, content=current_chunk.strip())
                )
                current_chunk = paragraph + "\n\n"
            elif len(paragraph) > chunk_size:
                # If the paragraph itself is too long, split it by sentences or just characters
                if current_chunk:
                    section_id += 1
                    sections.append(
                        Section(id=f"s{section_id}", section_type=SectionType.PARAGRAPH, content=current_chunk.strip())
                    )
                    current_chunk = ""

                # Try to split by sentences first
                sentences = paragraph.replace(". ", ".\n").split("\n")
                current_sentence_chunk = ""

                for sentence in sentences:
                    if len(current_sentence_chunk) + len(sentence) + 1 > chunk_size:
                        if current_sentence_chunk:
                            section_id += 1
                            sections.append(
                                Section(
                                    id=f"s{section_id}",
                                    section_type=SectionType.PARAGRAPH,
                                    content=current_sentence_chunk.strip(),
                                )
                            )
                            current_sentence_chunk = sentence + " "
                        else:
                            # Even a single sentence is too long, so split by characters
                            for i in range(0, len(sentence), chunk_size):
                                chunk = sentence[i : i + chunk_size]
                                section_id += 1
                                sections.append(
                                    Section(
                                        id=f"s{section_id}", section_type=SectionType.PARAGRAPH, content=chunk.strip()
                                    )
                                )
                    else:
                        current_sentence_chunk += sentence + " "

                if current_sentence_chunk:
                    section_id += 1
                    sections.append(
                        Section(
                            id=f"s{section_id}",
                            section_type=SectionType.PARAGRAPH,
                            content=current_sentence_chunk.strip(),
                        )
                    )
            else:
                # This paragraph fits, add it to the current chunk
                current_chunk += paragraph + "\n\n"

        # Add the last chunk if it's not empty
        if current_chunk.strip():
            section_id += 1
            sections.append(
                Section(id=f"s{section_id}", section_type=SectionType.PARAGRAPH, content=current_chunk.strip())
            )

        logger.debug(f"Created {len(sections)} fixed-size chunks")
        return sections
