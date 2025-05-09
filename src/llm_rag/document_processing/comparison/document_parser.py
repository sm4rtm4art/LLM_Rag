"""Module for parsing structured documents into logical sections."""

import re
from pathlib import Path
from typing import List, Optional, Union

from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

from .component_protocols import IParser
from .domain_models import DocumentFormat, InputChunk, ParserConfig, Section, SectionType

logger = get_logger(__name__)


class DocumentParser(IParser):
    """Parses documents into structured sections based on various strategies."""

    def __init__(
        self,
        config: Optional[ParserConfig] = None,
        default_format: DocumentFormat = DocumentFormat.MARKDOWN,
    ):
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
            f'Initialized DocumentParser with split_on_headings={self.config.split_on_headings}, '
            f'min_section_length={self.config.min_section_length}'
        )

    def parse(
        self, document: Union[str, Path, List[InputChunk]], format_type: Optional[DocumentFormat] = None
    ) -> List[Section]:
        """Parse document into sections.

        Args:
            document: Doc content (str, path, or List[InputChunk]).
            format_type: Format of the document (markdown, json, etc.).
                If None, will try to guess the format or use default.

        Returns:
            List of document sections.

        Raises:
            DocumentProcessingError: If parsing fails.

        """
        try:
            doc_format = format_type or self.default_format
            logger.debug(f'Parsing document with format: {doc_format.value}')

            # Handle PRE_CHUNKED format first
            if doc_format == DocumentFormat.PRE_CHUNKED:
                if isinstance(document, list):
                    # Optional: Add a check here to ensure all items in the list are InputChunk instances
                    # if not all(isinstance(chunk, InputChunk) for chunk in document):
                    #     raise DocumentProcessingError(
                    #         "Document is list but items are not InputChunk objects for PRE_CHUNKED format."
                    #     )
                    return self.parse_from_input_chunks(document)  # type: ignore
                else:
                    raise DocumentProcessingError(
                        f"For PRE_CHUNKED format, 'document' must be a list of InputChunk objects, "
                        f'got {type(document)}.'
                    )

            doc_content_str: str
            if isinstance(document, Path):
                doc_content_str = self._load_document(document)
            elif isinstance(document, str):
                # Heuristic to check if string is a path or content
                # This could be improved, e.g. by trying to Path(document).exists() if it looks like a path
                is_likely_path = (
                    (Path(document).suffix != '' and Path(document).is_file())
                    if len(document) < 260 and '\n' not in document
                    else False
                )
                if is_likely_path:
                    try:
                        doc_content_str = self._load_document(Path(document))
                    except DocumentProcessingError:  # If loading fails, treat as content
                        doc_content_str = document
                else:
                    doc_content_str = document
            else:
                # Consider if this exception type is the most appropriate for a failed type check.
                # TypeError more specific if 'document' isn't expected type *before* format dispatch.
                raise TypeError(
                    "Input 'document' must be a string, Path, or List[InputChunk] for non-PRE_CHUNKED formats."
                )

            if doc_format == DocumentFormat.MARKDOWN:
                sections = self._parse_markdown(doc_content_str)
            elif doc_format == DocumentFormat.JSON:
                sections = self._parse_json(doc_content_str)
            elif doc_format == DocumentFormat.TEXT:
                sections = self._parse_text(doc_content_str)
            else:
                # This case should ideally not be reached if PRE_CHUNKED is handled above
                # and other doc_format values are valid.
                raise ValueError(f'Unsupported document format: {doc_format}')

            logger.debug(f'Parsed {len(sections)} sections from document')
            return sections

        except Exception as e:
            error_msg = f'Error parsing document: {str(e)}'
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
        lines = markdown_content.splitlines()
        i = 0
        in_code_block = False
        code_block_lang = ''
        code_start_line_idx = -1

        parent_stack: List[Section] = []  # To track parent sections for nesting

        while i < len(lines):
            line = lines[i]

            # Handle code blocks
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_block_lang = line.strip()[3:].strip()
                    code_start_line_idx = i + 1
                    i += 1
                    continue
                else:
                    in_code_block = False
                    code_content_lines = lines[code_start_line_idx:i]
                    code_content = '\n'.join(code_content_lines)
                    current_parent = parent_stack[-1] if parent_stack else None
                    new_section = Section(
                        title='Code Block',
                        content=code_content,
                        level=current_parent.level + 1 if current_parent else 0,
                        section_type=SectionType.CODE,
                        metadata={'language': code_block_lang} if code_block_lang else {},
                        parent=current_parent,
                    )
                    sections.append(new_section)
                    if current_parent:
                        current_parent.children.append(new_section)
                    i += 1
                    continue

            if in_code_block:
                i += 1
                continue

            # Check for headings
            heading_match = None
            heading_level = 0
            for pattern in self._heading_patterns:
                match = pattern.match(line)
                if match:
                    # Determine level (ATX: count #, Setext: based on pattern)
                    if match.re.pattern == r'^#{1,6}\s+(.+)$':
                        heading_level = len(match.group(0).split(' ')[0])  # Count of #
                        heading_text = match.group(1).strip()
                    elif match.re.pattern == r'^(.+)\n[=]{2,}$':
                        # This pattern needs next line, handle multiline match if current line is part of it
                        if i + 1 < len(lines) and lines[i + 1].strip().startswith('==='):
                            heading_level = 1
                            heading_text = line.strip()
                            line = lines[i] + '\n' + lines[i + 1]  # Consume next line for this match
                            i += 1  # consume the underline
                        else:
                            continue  # Not a full Setext H1 match
                    elif match.re.pattern == r'^(.+)\n[-]{2,}$':
                        if i + 1 < len(lines) and lines[i + 1].strip().startswith('---'):
                            heading_level = 2
                            heading_text = line.strip()
                            line = lines[i] + '\n' + lines[i + 1]
                            i += 1  # consume the underline
                        else:
                            continue  # Not a full Setext H2 match
                    else:  # Should not happen if patterns are correct
                        heading_text = match.group(1).strip()
                        heading_level = 0  # Unknown heading level

                    heading_match = True  # Simplified, actual match object not used further directly
                    break

            current_parent = parent_stack[-1] if parent_stack else None

            if heading_match and self.config.split_on_headings:
                # Adjust parent stack based on heading level
                while parent_stack and parent_stack[-1].level >= heading_level:
                    parent_stack.pop()

                current_parent = parent_stack[-1] if parent_stack else None
                new_section = Section(
                    title=heading_text,
                    content='',  # Heading content is its title
                    level=heading_level,
                    section_type=SectionType.HEADING,
                    parent=current_parent,
                )
                sections.append(new_section)
                if current_parent:
                    current_parent.children.append(new_section)
                parent_stack.append(new_section)
                i += 1
                continue

            # Check for list items (simplified)
            is_list_item = line.strip().startswith(('*', '-', '+')) or re.match(r'^\s*\d+\.\s+', line)
            if is_list_item:
                list_content_lines = []
                while i < len(lines) and (
                    lines[i].strip().startswith(('*', '-', '+')) or re.match(r'^\s*\d+\.\s+', lines[i])
                ):
                    list_content_lines.append(lines[i].strip())
                    i += 1
                list_content = '\n'.join(list_content_lines)
                if list_content and len(list_content) >= self.config.min_section_length:
                    current_parent_for_list = parent_stack[-1] if parent_stack else None
                    new_section = Section(
                        title='List',
                        content=list_content,
                        level=current_parent_for_list.level + 1 if current_parent_for_list else 0,
                        section_type=SectionType.LIST,
                        parent=current_parent_for_list,
                    )
                    sections.append(new_section)
                    if current_parent_for_list:
                        current_parent_for_list.children.append(new_section)
                continue  # i is already advanced

            # Paragraphs (collect lines until next block element or empty lines if split_on_blank_lines)
            paragraph_lines = []
            start_paragraph_line = i
            while i < len(lines):
                current_line_text = lines[i]
                is_next_heading = any(p.match(current_line_text) for p in self._heading_patterns)
                is_next_list_item = current_line_text.strip().startswith(('*', '-', '+')) or re.match(
                    r'^\s*\d+\.\s+', current_line_text
                )
                is_code_block_start = current_line_text.strip().startswith('```')

                if is_next_heading or is_next_list_item or is_code_block_start:
                    break
                if self.config.split_on_blank_lines and not current_line_text.strip():
                    i += 1  # Consume blank line and break
                    break

                paragraph_lines.append(current_line_text)
                i += 1

            paragraph_content = '\n'.join(paragraph_lines).strip()
            if paragraph_content and len(paragraph_content) >= self.config.min_section_length:
                parent_for_paragraph = parent_stack[-1] if parent_stack else None
                new_section = Section(
                    title='Paragraph',
                    content=paragraph_content,
                    level=parent_for_paragraph.level + 1 if parent_for_paragraph else 0,
                    section_type=SectionType.PARAGRAPH,
                    parent=parent_for_paragraph,
                )
                sections.append(new_section)
                if parent_for_paragraph:
                    parent_for_paragraph.children.append(new_section)
            elif (
                not paragraph_lines and i == start_paragraph_line
            ):  # Avoid infinite loop on empty line if not splitting
                i += 1

        return sections

    def _parse_json(self, json_content: str) -> List[Section]:
        """Parse JSON document into sections.

        Args:
            json_content: The JSON document as a string.

        Returns:
            List of document sections.

        """
        import json

        data = json.loads(json_content)
        sections: List[Section] = []

        def _extract_sections_recursive(json_node, level, parent_section=None):
            if isinstance(json_node, dict):
                # Handle top-level title specifically as a main heading if it's the root call
                is_root_call = level == 0 and parent_section is None and json_node is data

                node_title_key = 'title' if 'title' in json_node else 'heading'
                node_title = str(json_node.get(node_title_key, 'Section'))
                node_content = str(json_node.get('content', ''))

                # Determine section type
                section_type_str = json_node.get('type', '').upper()
                if is_root_call and 'title' in json_node and not section_type_str:
                    # If it's the root and has a 'title' but no explicit 'type', make it a HEADING
                    current_section_type = SectionType.HEADING
                elif section_type_str in SectionType.__members__:
                    current_section_type = SectionType[section_type_str]
                else:
                    current_section_type = SectionType.PARAGRAPH if node_content else SectionType.UNKNOWN

                # If it's a heading type and content is empty, use title as content (or keep content empty)
                # For this parser, a HEADING Section has its text in 'title',
                # and 'content' can be empty or for subsequent paras.
                # The refactored markdown parser sets heading content to "".
                display_content = node_content
                if current_section_type == SectionType.HEADING and not node_content:
                    pass  # Title is already set, content can remain empty for heading objects

                current_section = Section(
                    title=node_title,
                    content=display_content,
                    level=level,
                    section_type=current_section_type,
                    parent=parent_section,
                )
                sections.append(current_section)
                if parent_section:
                    parent_section.children.append(current_section)

                children_nodes = json_node.get('children', json_node.get('sections', []))
                if isinstance(children_nodes, list):
                    for child_node in children_nodes:
                        _extract_sections_recursive(child_node, level + 1, current_section)
                elif isinstance(children_nodes, dict):  # Handle if children/sections is a single dict node
                    _extract_sections_recursive(children_nodes, level + 1, current_section)

            elif isinstance(json_node, list):
                # If the root itself is a list of sections
                for _, item in enumerate(json_node):
                    # Pass parent_section to maintain hierarchy if list items are children of a conceptual parent
                    _extract_sections_recursive(item, level, parent_section)

        _extract_sections_recursive(data, 0)

        if not sections and json_content:  # Fallback for completely unhandled JSON structure
            sections.append(
                Section(title='JSON Content', content=json_content, level=0, section_type=SectionType.UNKNOWN)
            )
        return sections

    def _parse_text(self, text_content: str) -> List[Section]:
        """Parse plain text document into sections.

        Args:
            text_content: The text document as a string.

        Returns:
            List of document sections.

        """
        sections = []
        paragraphs = re.split(r'\n\s*\n', text_content)
        for para_content in paragraphs:
            content = para_content.strip()
            if content and len(content) >= self.config.min_section_length:
                sections.append(
                    Section(title='Paragraph', content=content, level=0, section_type=SectionType.PARAGRAPH)
                )
        return sections

    def _load_document(self, document_path: Path) -> str:
        """Load a document from a file path.

        Args:
            document_path: Path to the document file.

        Returns:
            The content of the document as a string.

        Raises:
            DocumentProcessingError: If the document cannot be loaded.

        """
        try:
            if not document_path.exists():
                raise DocumentProcessingError(f'Document file not found: {document_path}')
            if not document_path.is_file():
                raise DocumentProcessingError(f'Document path is not a file: {document_path}')
            content = document_path.read_text(encoding='utf-8')
            logger.debug(f'Loaded document {document_path} with {len(content)} characters')
            return content
        except Exception as e:
            error_msg = f'Error loading document from {document_path}: {str(e)}'
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
        logger.debug(f'Segmenting text into fixed chunks of size {chunk_size}')
        sections = []
        if not isinstance(content, str):
            content = str(content)

        if len(content) <= chunk_size:
            return [Section(title='Chunk', content=content.strip(), level=0, section_type=SectionType.PARAGRAPH)]

        paragraphs = content.split('\n\n')
        current_chunk_content = []
        current_chunk_len = 0

        for paragraph in paragraphs:
            para_len = len(paragraph)
            if current_chunk_len + para_len + (1 if current_chunk_content else 0) > chunk_size:
                if current_chunk_content:
                    sections.append(
                        Section(
                            title='Chunk',
                            content='\n\n'.join(current_chunk_content).strip(),
                            level=0,
                            section_type=SectionType.PARAGRAPH,
                        )
                    )
                    current_chunk_content = []
                    current_chunk_len = 0

                # If paragraph itself is too long, split it
                if para_len > chunk_size:
                    for i in range(0, para_len, chunk_size):
                        sections.append(
                            Section(
                                title='Chunk',
                                content=paragraph[i : i + chunk_size].strip(),
                                level=0,
                                section_type=SectionType.PARAGRAPH,
                            )
                        )
                    continue  # Move to next paragraph

            current_chunk_content.append(paragraph)
            current_chunk_len += para_len + (1 if len(current_chunk_content) > 1 else 0)  # +1 for newline joiner

        if current_chunk_content:  # Add any remaining chunk
            sections.append(
                Section(
                    title='Chunk',
                    content='\n\n'.join(current_chunk_content).strip(),
                    level=0,
                    section_type=SectionType.PARAGRAPH,
                )
            )

        logger.debug(f'Created {len(sections)} fixed-size chunks')
        return sections

    def parse_from_input_chunks(self, chunks: List[InputChunk]) -> List[Section]:
        """Parse a list of InputChunk objects into a list of Section objects.

        Args:
            chunks: A list of InputChunk objects, typically from a semantic chunker.

        Returns:
            A list of Section objects.

        """
        logger.debug(f'Parsing {len(chunks)} pre-defined input chunks.')
        parsed_sections: List[Section] = []
        for i, chunk in enumerate(chunks):
            title = chunk.metadata.get(self.config.chunk_title_metadata_key, f'Segment {i + 1}')
            # A more robust title strategy could be implemented here
            # e.g. if no title in metadata, check first line of content for heading-like structure

            level = chunk.metadata.get(self.config.chunk_level_metadata_key, self.config.default_chunk_level)
            if not isinstance(level, int):
                level = self.config.default_chunk_level

            section_type_str = chunk.metadata.get(self.config.chunk_type_metadata_key)
            current_section_type = self.config.default_chunk_section_type
            if section_type_str and isinstance(section_type_str, str):
                try:
                    current_section_type = SectionType[section_type_str.upper()]
                except KeyError:
                    logger.warning(f"Unknown section type '{section_type_str}' in chunk metadata. Defaulting.")

            # For semantic chunks, parent/children are typically not pre-defined unless chunker provides hierarchy
            new_section = Section(
                title=str(title),  # Ensure title is a string
                content=chunk.content,
                level=int(level),  # Ensure level is an int
                section_type=current_section_type,
                metadata=chunk.metadata.copy() if chunk.metadata else {},  # Pass along metadata
                parent=None,  # Assuming flat list from chunker for now
                children=[],  # Assuming flat list
            )
            parsed_sections.append(new_section)

        logger.info(f'Successfully parsed {len(parsed_sections)} sections from input chunks.')
        return parsed_sections
