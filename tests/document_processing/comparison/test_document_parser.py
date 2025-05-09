"""Unit tests for the document parser."""

import json
import unittest
from pathlib import Path
from unittest import mock

from llm_rag.document_processing.comparison.document_parser import DocumentParser  # Implementation

# Updated imports to use domain_models
from llm_rag.document_processing.comparison.domain_models import (
    DocumentFormat,
    ParserConfig,  # ParserConfig might be needed if tests instantiate parser with it
    Section,
    SectionType,
)
from llm_rag.utils.errors import DocumentProcessingError


class TestDocumentParser(unittest.TestCase):
    """Test cases for DocumentParser."""

    def setUp(self):
        """Set up test cases."""
        self.parser = DocumentParser()  # Uses default ParserConfig from domain_models
        self.markdown_content = """# Heading 1

This is paragraph 1 under heading 1.

This is paragraph 2 under heading 1.

## Heading 2

- List item 1
- List item 2

### Heading 3

```
Code block example
def hello():
    print("Hello")
```
"""
        self.json_content = json.dumps(
            {
                'title': 'Sample Document',
                'sections': [
                    {'heading': 'Section 1', 'content': 'This is section 1 content.'},
                    {
                        'heading': 'Section 2',
                        'content': 'This is section 2 content.',
                        # Note: the parser's _parse_json was updated to handle 'children' or 'sections' keys
                        'children': [{'heading': 'Subsection 2.1', 'content': 'This is subsection content.'}],
                    },
                ],
            }
        )
        self.text_content = """This is paragraph 1.

This is paragraph 2.

This is paragraph 3."""

    def test_init(self):
        parser = DocumentParser()
        self.assertEqual(parser.config.default_format, DocumentFormat.MARKDOWN)
        parser_json = DocumentParser(config=ParserConfig(default_format=DocumentFormat.JSON))
        self.assertEqual(parser_json.config.default_format, DocumentFormat.JSON)

    def test_parse_markdown(self):
        """Test parsing markdown content."""
        # Using format_type instead of format
        sections = self.parser.parse(self.markdown_content, format_type=DocumentFormat.MARKDOWN)
        self.assertIsInstance(sections, list)
        self.assertGreater(len(sections), 0)
        headings = [s for s in sections if s.section_type == SectionType.HEADING]
        self.assertEqual(len(headings), 3)
        heading_levels = [h.level for h in headings]
        self.assertEqual(heading_levels, [1, 2, 3])
        lists = [s for s in sections if s.section_type == SectionType.LIST]
        self.assertGreater(len(lists), 0)
        code_blocks = [s for s in sections if s.section_type == SectionType.CODE]
        self.assertGreater(len(code_blocks), 0)
        # Content of heading is in title field now
        self.assertEqual(headings[0].title, 'Heading 1')

    def test_parse_json(self):
        """Test parsing JSON content."""
        sections = self.parser.parse(self.json_content, format_type=DocumentFormat.JSON)
        self.assertIsInstance(sections, list)
        self.assertGreater(len(sections), 0)
        title_found = False
        section1_found = False
        subsection_found = False
        for section in sections:
            if section.section_type == SectionType.HEADING and section.title == 'Sample Document':
                title_found = True
            if section.title == 'Section 1' and 'This is section 1 content.' in section.content:
                section1_found = True
            if section.title == 'Subsection 2.1' and 'This is subsection content.' in section.content:
                subsection_found = True
        self.assertTrue(title_found, 'Document title not found or parsed incorrectly')
        self.assertTrue(section1_found, 'Section 1 not found or parsed incorrectly')
        self.assertTrue(subsection_found, 'Subsection 2.1 not found or parsed incorrectly')

    def test_parse_text(self):
        """Test parsing plain text content."""
        sections = self.parser.parse(self.text_content, format_type=DocumentFormat.TEXT)
        self.assertIsInstance(sections, list)
        self.assertEqual(len(sections), 3)
        for section in sections:
            self.assertEqual(section.section_type, SectionType.PARAGRAPH)
            self.assertEqual(section.title, 'Paragraph')  # Default title for paragraphs from parser
        self.assertIn('paragraph 1', sections[0].content)
        self.assertIn('paragraph 2', sections[1].content)
        self.assertIn('paragraph 3', sections[2].content)

    @mock.patch('pathlib.Path.exists')
    def test_parse_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        non_existent_file = Path('non_existent_file.md')
        with self.assertRaises(DocumentProcessingError):
            # parse expects content or a resolvable path. If Path('...').exists() is False,
            # it might treat it as content unless logic in parse is very specific.
            # The DocumentParser.parse was updated to try _load_document if it looks like a path.
            # _load_document then raises if Path.exists() is false.
            self.parser.parse(non_existent_file, format_type=DocumentFormat.MARKDOWN)

    # Updated mock for Path.read_text()
    @mock.patch('pathlib.Path.read_text', return_value='# Test Heading\n\nTest paragraph.')
    @mock.patch('pathlib.Path.is_file', return_value=True)
    @mock.patch('pathlib.Path.exists', return_value=True)
    def test_parse_file(self, mock_exists, mock_is_file, mock_read_text):
        test_file = Path('test.md')
        sections = self.parser.parse(test_file, format_type=DocumentFormat.MARKDOWN)
        self.assertIsInstance(sections, list)
        # The new parser is more detailed, check actual output or simplify test assertion
        # Based on refactored _parse_markdown, a heading and its content might be separate sections or nested.
        # For "# Test Heading\n\nTest paragraph.":
        # 1. Section(title='Test Heading', content='', level=1, type=HEADING)
        # 2. Section(title='Paragraph', content='Test paragraph.', level=2 (child of heading), type=PARAGRAPH)
        # Or if not nested directly: count might be 2, one heading one paragraph
        # Let's check for presence and types/titles
        self.assertTrue(any(s.section_type == SectionType.HEADING and s.title == 'Test Heading' for s in sections))
        self.assertTrue(
            any(s.section_type == SectionType.PARAGRAPH and s.content == 'Test paragraph.' for s in sections)
        )
        mock_read_text.assert_called_once_with(encoding='utf-8')

    def test_segment_by_fixed_chunks(self):
        """Test segmenting text into fixed-size chunks."""
        long_text = 'This is a test. ' * 100  # About 1500 characters

        chunks = self.parser.segment_by_fixed_chunks(long_text, chunk_size=500)

        # Assertions
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks

        # Each chunk should be a Section object
        for chunk in chunks:
            self.assertIsInstance(chunk, Section)
            self.assertEqual(chunk.section_type, SectionType.PARAGRAPH)

            # Each chunk should be roughly at most 500 characters
            # Allow some margin because we split on line boundaries
            self.assertLessEqual(len(chunk.content), 550)


def test_section_hashability():  # This is a standalone function test, not part of TestDocumentParser class
    """Test that Section objects are hashable and can be used in sets and dictionaries."""
    # Using new Section constructor parameters
    section1 = Section(
        title='Test1', content='Test content', level=0, section_type=SectionType.PARAGRAPH, section_id='test1'
    )
    section2 = Section(
        title='Test1', content='Test content', level=0, section_type=SectionType.PARAGRAPH, section_id='test1'
    )
    section3 = Section(
        title='Test2', content='Test content', level=0, section_type=SectionType.PARAGRAPH, section_id='test2'
    )
    section_set = {section1, section2, section3}
    assert len(section_set) == 2
    section_dict = {section1: 'value1', section3: 'value3'}
    assert len(section_dict) == 2
    assert section_dict[section2] == 'value1'
    assert hash(section1) != hash(section3)
    assert hash(section1) == hash(section2)


if __name__ == '__main__':
    unittest.main()
