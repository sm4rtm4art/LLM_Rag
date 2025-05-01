import unittest

from llm_rag.document_processing.ocr.output_formatter import JSONFormatter, MarkdownFormatter


class TestMarkdownFormatter(unittest.TestCase):
    """Test cases for the MarkdownFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = MarkdownFormatter()

    def test_format_document_empty_input(self):
        """Test formatting an empty document."""
        # Test with empty list
        result = self.formatter.format_document([])
        self.assertEqual(result, "")

        # Test with empty dictionary
        result = self.formatter.format_document({})
        self.assertEqual(result, "")

    def test_format_document_with_list(self):
        """Test formatting a document from a list of pages."""
        pages = ["Page 1 content", "Page 2 content", "Page 3 content"]
        result = self.formatter.format_document(pages)

        # Verify first page doesn't have page marker
        self.assertTrue(result.startswith("Page 1 content"))

        # Verify other pages have page markers
        self.assertIn("## Page 2", result)
        self.assertIn("## Page 3", result)

    def test_format_document_with_dict(self):
        """Test formatting a document from a dictionary of pages."""
        pages = {1: "Page 1 content", 2: "Page 2 content", 3: "Page 3 content"}
        result = self.formatter.format_document(pages)

        # Verify first page doesn't have page marker
        self.assertTrue(result.startswith("Page 1 content"))

        # Verify other pages have page markers
        self.assertIn("## Page 2", result)
        self.assertIn("## Page 3", result)

    def test_format_page_empty_input(self):
        """Test formatting an empty page."""
        result = self.formatter.format_page("")
        self.assertEqual(result, "")

    def test_format_page_paragraphs(self):
        """Test formatting a page with multiple paragraphs."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = self.formatter.format_page(text)

        # Verify paragraphs are preserved
        self.assertEqual(result, "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")

    def test_heading_detection(self):
        """Test detection of headings."""
        # These should be detected as headings
        headings = ["CHAPTER 1", "1. Introduction", "1.1 Background", "Section A"]

        for heading in headings:
            self.assertTrue(self.formatter._is_heading(heading), f"Failed to detect heading: {heading}")

        # These should not be detected as headings
        non_headings = [
            "This is a normal paragraph that ends with a period.",
            "This text is too long to be a heading as headings are typically short and "
            "don't contain too much information like this sentence does.",
        ]

        for text in non_headings:
            self.assertFalse(self.formatter._is_heading(text), f"Incorrectly detected as heading: {text}")

    def test_list_detection(self):
        """Test detection of list items."""
        # These should be detected as list items
        list_items = ["• First item", "- Second item", "* Third item", "1. Numbered item", "a) Lettered item"]

        for item in list_items:
            self.assertTrue(self.formatter._is_list_item(item), f"Failed to detect list item: {item}")

        # These should not be detected as list items
        non_list_items = ["Normal paragraph", "Text with a - in the middle"]

        for text in non_list_items:
            self.assertFalse(self.formatter._is_list_item(text), f"Incorrectly detected as list item: {text}")


class TestJSONFormatter(unittest.TestCase):
    """Test cases for the JSONFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = JSONFormatter()

    def test_format_document_list(self):
        """Test formatting a document from a list."""
        pages = ["Page 1", "Page 2"]
        result = self.formatter.format_document(pages)

        self.assertEqual(result["pages"]["0"]["text"], "Page 1")
        self.assertEqual(result["pages"]["1"]["text"], "Page 2")

    def test_format_document_dict(self):
        """Test formatting a document from a dictionary."""
        pages = {1: "Page 1", 2: "Page 2"}
        result = self.formatter.format_document(pages)

        self.assertEqual(result["pages"]["1"]["text"], "Page 1")
        self.assertEqual(result["pages"]["2"]["text"], "Page 2")


if __name__ == "__main__":
    unittest.main()
