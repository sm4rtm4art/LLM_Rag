"""Module for parsing documents into logical sections for comparison."""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentFormat(Enum):
    """Supported document formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


class SectionType(Enum):
    """Types of document sections."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    TABLE = "table"
    OTHER = "other"


class Section:
    """Represents a logical section of a document.

    A section is a discrete part of a document with its own semantic meaning,
    such as a heading, paragraph, list, or code block.
    """

    def __init__(
        self,
        content: str,
        section_type: SectionType = SectionType.PARAGRAPH,
        level: int = 0,
        metadata: Optional[Dict] = None,
    ):
        """Initialize a document section.

        Args:
            content: The text content of the section.
            section_type: The type of section (heading, paragraph, etc.).
            level: The nesting level of the section (e.g., heading level).
            metadata: Additional metadata about the section.

        """
        self.content = content
        self.section_type = section_type
        self.level = level
        self.metadata = metadata or {}

    def __str__(self) -> str:
        """Return a string representation of the section."""
        return f"{self.section_type.value.capitalize()} (L{self.level}): {self.content[:50]}..."

    def __repr__(self) -> str:
        """Return a developer representation of the section."""
        return (
            f"Section(type={self.section_type.value}, level={self.level}, "
            f"content='{self.content[:30]}...', metadata={self.metadata})"
        )


class DocumentParser:
    """Parser for structured documents.

    This class is responsible for parsing structured documents (Markdown, JSON)
    into logical sections based on headings, semantic breaks, or fixed chunks.
    """

    def __init__(self, default_format: DocumentFormat = DocumentFormat.MARKDOWN):
        """Initialize the document parser.

        Args:
            default_format: The default format to use when not specified.

        """
        self.default_format = default_format
        logger.info(f"Initialized DocumentParser with default format: {default_format.value}")

    def parse(self, document: Union[str, Path], format: Optional[DocumentFormat] = None) -> List[Section]:
        """Parse a document into a list of logical sections.

        Args:
            document: The document content or path to the document file.
            format: The format of the document. If None, will attempt to detect.

        Returns:
            A list of Section objects representing the document structure.

        Raises:
            DocumentProcessingError: If the document cannot be parsed.

        """
        # Determine format if not provided
        doc_format = format or self.default_format

        # Determine if document is a file path or content
        content = document

        # If it's a Path object, treat it as a file path
        if isinstance(document, Path):
            path = document
            try:
                if path.exists():
                    if path.is_file():
                        content = self._load_document(path)
                    else:
                        error_msg = f"Document path is not a file: {document}"
                        logger.error(error_msg)
                        raise DocumentProcessingError(error_msg)
                else:
                    error_msg = f"Document file not found: {document}"
                    logger.error(error_msg)
                    raise DocumentProcessingError(error_msg)
            except OSError as e:
                error_msg = f"Error accessing document file: {document}: {str(e)}"
                logger.error(error_msg)
                raise DocumentProcessingError(error_msg) from e
        # For strings, check if it looks like a file path
        elif isinstance(document, str):
            # Heuristics to determine if this is a file path:
            # 1. No newlines (content usually has newlines)
            # 2. Less than 255 chars (typical max path length)
            # 3. Contains file extension or path separators
            # 4. Doesn't start with common content markers
            is_likely_path = (
                "\n" not in document
                and len(document) < 255
                and ("." in document or "/" in document or "\\" in document)
                and not document.lstrip().startswith(("#", "{", "[", "<", "-"))
            )

            if is_likely_path:
                path = Path(document)
                try:
                    if path.exists():
                        if path.is_file():
                            content = self._load_document(path)
                        else:
                            error_msg = f"Document path is not a file: {document}"
                            logger.error(error_msg)
                            raise DocumentProcessingError(error_msg)
                    else:
                        error_msg = f"Document file not found: {document}"
                        logger.error(error_msg)
                        raise DocumentProcessingError(error_msg)
                except OSError:
                    # OSError from path operations - likely not a valid path
                    # Just treat as content
                    pass

        try:
            # Parse based on format
            if doc_format == DocumentFormat.MARKDOWN:
                return self._parse_markdown(content)
            elif doc_format == DocumentFormat.JSON:
                return self._parse_json(content)
            elif doc_format == DocumentFormat.TEXT:
                return self._parse_text(content)
            else:
                raise ValueError(f"Unsupported document format: {doc_format}")

        except DocumentProcessingError:
            # Re-raise DocumentProcessingError
            raise
        except Exception as e:
            error_msg = f"Error parsing document: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

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

    def _parse_markdown(self, content: str) -> List[Section]:
        """Parse Markdown content into logical sections.

        Args:
            content: Markdown content as a string.

        Returns:
            A list of Section objects representing the Markdown structure.

        """
        logger.debug("Parsing markdown content")
        sections = []
        current_section = None
        current_text = ""
        current_level = 0

        # Make sure content is a string
        if not isinstance(content, str):
            content = str(content)

        # Split content into lines for processing
        lines = content.splitlines()

        for line in lines:
            # Check for heading
            if line.startswith("#"):
                # Save previous section if exists
                if current_text and current_section:
                    sections.append(Section(current_text.strip(), current_section, current_level))
                    current_text = ""

                # Determine heading level
                level = len(line) - len(line.lstrip("#"))
                heading_text = line[level:].strip().lstrip()

                # Add heading as a section
                sections.append(Section(heading_text, SectionType.HEADING, level))

                # Reset current section tracking
                current_section = SectionType.PARAGRAPH
                current_level = 0

            # Check for code block
            elif line.startswith("```"):
                if current_section == SectionType.CODE:
                    # End of code block
                    sections.append(Section(current_text.strip(), SectionType.CODE, 0))
                    current_text = ""
                    current_section = SectionType.PARAGRAPH
                    current_level = 0
                else:
                    # Start of code block, save previous section if exists
                    if current_text and current_section:
                        sections.append(Section(current_text.strip(), current_section, current_level))
                        current_text = ""

                    current_section = SectionType.CODE
                    current_level = 0

            # Process bullet list items
            elif line.strip().startswith(("- ", "* ", "+ ")):
                if current_section != SectionType.LIST:
                    # Save previous section if exists
                    if current_text and current_section:
                        sections.append(Section(current_text.strip(), current_section, current_level))
                        current_text = ""

                    current_section = SectionType.LIST
                    current_level = 0

                current_text += line + "\n"

            # Handle paragraph breaks
            elif not line.strip() and current_text and current_section:
                # Empty line - save current section and start a new one
                sections.append(Section(current_text.strip(), current_section, current_level))
                current_text = ""
                current_section = SectionType.PARAGRAPH
                current_level = 0

            # Add to current section
            elif line.strip() or current_text:
                if not current_section:
                    current_section = SectionType.PARAGRAPH
                    current_level = 0

                current_text += line + "\n"

        # Add the last section if there's one pending
        if current_text and current_section:
            sections.append(Section(current_text.strip(), current_section, current_level))

        logger.debug(f"Parsed {len(sections)} sections from markdown")
        return sections

    def _parse_json(self, content: str) -> List[Section]:
        """Parse JSON content into logical sections.

        Args:
            content: JSON content as a string.

        Returns:
            A list of Section objects representing the JSON structure.

        """
        logger.debug("Parsing JSON content")

        # Ensure we have a string
        if not isinstance(content, str):
            content = str(content)

        import json

        try:
            # Parse JSON
            data = json.loads(content)
            sections = []

            # Process title directly if available
            if isinstance(data, dict) and "title" in data:
                sections.append(Section(str(data["title"]), SectionType.HEADING, 1))

            # Process sections directly if available
            if isinstance(data, dict) and "sections" in data and isinstance(data["sections"], list):
                for section in data["sections"]:
                    if isinstance(section, dict):
                        # Add heading
                        if "heading" in section:
                            sections.append(Section(section["heading"], SectionType.HEADING, 2))
                        # Add content
                        if "content" in section:
                            sections.append(Section(section["content"], SectionType.PARAGRAPH, 0))
                        # Handle subsections
                        if "subsections" in section and isinstance(section["subsections"], list):
                            for subsection in section["subsections"]:
                                if isinstance(subsection, dict):
                                    if "heading" in subsection:
                                        sections.append(Section(subsection["heading"], SectionType.HEADING, 3))
                                    if "content" in subsection:
                                        sections.append(Section(subsection["content"], SectionType.PARAGRAPH, 0))

            # If sections list is still empty, use the generic conversion
            if not sections:
                sections = self._convert_json_to_sections(data)

            logger.debug(f"Parsed {len(sections)} sections from JSON")
            return sections
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def _convert_json_to_sections(self, data: Union[Dict, List], prefix="", level=0) -> List[Section]:
        """Convert a JSON object to sections recursively.

        Args:
            data: JSON data (dict or list).
            prefix: Key prefix for nested data.
            level: Current nesting level.

        Returns:
            List of Section objects representing the JSON structure.

        """
        sections = []

        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value, (dict, list)):
                    # Add the key as a heading
                    sections.append(Section(full_key, SectionType.HEADING, level + 1))
                    # Recursively process nested structures
                    sections.extend(self._convert_json_to_sections(value, full_key, level + 1))
                else:
                    # Add leaf node as a section
                    content = f"{full_key}: {value}"
                    sections.append(Section(content, SectionType.PARAGRAPH, level))

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    # Add list index as pseudo-heading
                    list_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
                    sections.append(Section(list_key, SectionType.HEADING, level + 1))
                    # Recursively process nested structures
                    sections.extend(self._convert_json_to_sections(item, list_key, level + 1))
                else:
                    # Add list item as a section
                    content = f"{prefix}[{i}]: {item}" if prefix else f"[{i}]: {item}"
                    sections.append(Section(content, SectionType.LIST, level))

        return sections

    def _parse_text(self, content: str) -> List[Section]:
        """Parse plain text content into logical sections.

        Args:
            content: Plain text content as a string.

        Returns:
            A list of Section objects representing the text structure.

        """
        logger.debug("Parsing plain text content")
        sections = []

        # Split by double newlines (paragraph breaks)
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        for paragraph in paragraphs:
            sections.append(Section(paragraph, SectionType.PARAGRAPH, 0))

        logger.debug(f"Parsed {len(sections)} sections from plain text")
        return sections

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
