"""Unit tests for the document parser."""

import json
import unittest
from pathlib import Path
from unittest import mock

from llm_rag.document_processing.comparison.document_parser import DocumentFormat, DocumentParser, Section, SectionType
from llm_rag.utils.errors import DocumentProcessingError


class TestDocumentParser(unittest.TestCase):
    """Test cases for DocumentParser."""

    def setUp(self):
        """Set up test cases."""
        self.parser = DocumentParser()

        # Sample markdown content
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

        # Sample JSON content
        self.json_content = json.dumps(
            {
                "title": "Sample Document",
                "sections": [
                    {"heading": "Section 1", "content": "This is section 1 content."},
                    {
                        "heading": "Section 2",
                        "content": "This is section 2 content.",
                        "subsections": [{"heading": "Subsection 2.1", "content": "This is subsection content."}],
                    },
                ],
            }
        )

        # Sample plain text content
        self.text_content = """This is paragraph 1.

This is paragraph 2.

This is paragraph 3."""

    def test_init(self):
        """Test initialization of DocumentParser."""
        # Test with default format
        parser = DocumentParser()
        self.assertEqual(parser.default_format, DocumentFormat.MARKDOWN)

        # Test with specific format
        parser = DocumentParser(default_format=DocumentFormat.JSON)
        self.assertEqual(parser.default_format, DocumentFormat.JSON)

    def test_parse_markdown(self):
        """Test parsing markdown content."""
        sections = self.parser.parse(self.markdown_content, format=DocumentFormat.MARKDOWN)

        # Assertions
        self.assertIsInstance(sections, list)
        self.assertGreater(len(sections), 0)

        # Check for heading sections
        headings = [s for s in sections if s.section_type == SectionType.HEADING]
        self.assertEqual(len(headings), 3)

        # Check heading levels
        heading_levels = [h.level for h in headings]
        self.assertEqual(heading_levels, [1, 2, 3])

        # Check for list section
        lists = [s for s in sections if s.section_type == SectionType.LIST]
        self.assertGreater(len(lists), 0)

        # Check for code section
        code_blocks = [s for s in sections if s.section_type == SectionType.CODE]
        self.assertGreater(len(code_blocks), 0)

        # Check content of first heading
        self.assertEqual(headings[0].content, "Heading 1")

    def test_parse_json(self):
        """Test parsing JSON content."""
        # Make sure the JSON content is treated as content, not a file path
        sections = self.parser.parse(self.json_content, format=DocumentFormat.JSON)

        # Assertions
        self.assertIsInstance(sections, list)
        self.assertGreater(len(sections), 0)

        # Check section content
        title_found = False
        section1_found = False

        for section in sections:
            if section.section_type == SectionType.HEADING and "Sample Document" in section.content:
                title_found = True
            if "This is section 1 content" in section.content:
                section1_found = True

        self.assertTrue(title_found)
        self.assertTrue(section1_found)

    def test_parse_text(self):
        """Test parsing plain text content."""
        sections = self.parser.parse(self.text_content, format=DocumentFormat.TEXT)

        # Assertions
        self.assertIsInstance(sections, list)
        self.assertEqual(len(sections), 3)  # 3 paragraphs

        # All sections should be paragraphs
        for section in sections:
            self.assertEqual(section.section_type, SectionType.PARAGRAPH)

        # Check content
        self.assertIn("paragraph 1", sections[0].content)
        self.assertIn("paragraph 2", sections[1].content)
        self.assertIn("paragraph 3", sections[2].content)

    @mock.patch("pathlib.Path.exists")
    def test_parse_file_not_found(self, mock_exists):
        """Test handling of non-existent file."""
        # Mock Path.exists to return False for this test
        mock_exists.return_value = False

        # Use a file path that will be recognized as a file path and not content
        non_existent_file = Path("non_existent_file.md")

        with self.assertRaises(DocumentProcessingError):
            self.parser.parse(non_existent_file)

    @mock.patch("builtins.open", mock.mock_open(read_data="# Test Heading\n\nTest paragraph."))
    @mock.patch("pathlib.Path.exists", return_value=True)
    @mock.patch("pathlib.Path.is_file", return_value=True)
    def test_parse_file(self, mock_is_file, mock_exists):
        """Test parsing a file."""
        # Use a file path that will be recognized as a file path and not content
        test_file = Path("test.md")
        sections = self.parser.parse(test_file)

        # Assertions
        self.assertIsInstance(sections, list)
        self.assertEqual(len(sections), 2)  # 1 heading, 1 paragraph

        # Check section types
        self.assertEqual(sections[0].section_type, SectionType.HEADING)
        self.assertEqual(sections[1].section_type, SectionType.PARAGRAPH)

        # Check content
        self.assertEqual(sections[0].content, "Test Heading")
        self.assertEqual(sections[1].content, "Test paragraph.")

    def test_segment_by_fixed_chunks(self):
        """Test segmenting text into fixed-size chunks."""
        long_text = "This is a test. " * 100  # About 1500 characters

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


if __name__ == "__main__":
    unittest.main()
