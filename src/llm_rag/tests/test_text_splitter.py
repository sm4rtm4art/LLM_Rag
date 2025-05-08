"""Tests for the TextSplitter class."""

import unittest
from typing import List

from src.llm_rag.document_processing.processors import TextSplitter


class TestTextSplitter(unittest.TestCase):
    """Tests for the TextSplitter class."""

    def test_init_with_separators(self):
        """Test initialization with separators list."""
        separators = ['\n\n', '\n', '.', ' ']
        splitter = TextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separators=separators,
        )
        self.assertEqual(splitter.separators, separators)

    def test_init_with_separator(self):
        """Test initialization with single separator (deprecated)."""
        separator = ';'
        # Use with statement to ignore the DeprecationWarning
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            splitter = TextSplitter(
                chunk_size=100,
                chunk_overlap=20,
                separator=separator,
            )
            # Check that a DeprecationWarning was raised
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w))
            self.assertTrue(any('deprecated' in str(warning.message) for warning in w))

        # Verify that the separator was converted to a list
        self.assertEqual(splitter.separators, [separator])

    def test_init_with_both_separator_and_separators(self):
        """Test initialization with both separator and separators parameters."""
        separator = ';'
        separators = ['\n\n', '\n', '.', ' ']
        # Use with statement to ignore the DeprecationWarning
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore')
            splitter = TextSplitter(
                chunk_size=100,
                chunk_overlap=20,
                separator=separator,
                separators=separators,
            )

        # Verify that separators takes precedence over separator
        self.assertEqual(splitter.separators, separators)

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        splitter = TextSplitter()
        # Default separators
        default_separators = ['\n\n', '\n', ' ', '']
        self.assertEqual(splitter.separators, default_separators)

    def test_split_text(self):
        """Test splitting text."""
        text = 'This is a test.\nWith multiple lines.\nAnd some more text.'
        splitter = TextSplitter(chunk_size=20, chunk_overlap=0)
        chunks = splitter.split_text(text)
        self.assertIsInstance(chunks, List)
        self.assertGreater(len(chunks), 1)
        # Each chunk should be at most 20 characters
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 20)

    def test_split_documents(self):
        """Test splitting documents."""
        documents = [
            {
                'content': 'This is a test document with some content.',
                'metadata': {'source': 'test1.txt'},
            },
            {
                'content': 'This is another test document with different content.',
                'metadata': {'source': 'test2.txt'},
            },
        ]
        splitter = TextSplitter(chunk_size=20, chunk_overlap=0)
        chunks = splitter.split_documents(documents)
        self.assertIsInstance(chunks, List)
        self.assertGreater(len(chunks), 2)  # Should have more chunks than original documents
        for chunk in chunks:
            self.assertIn('content', chunk)
            self.assertIn('metadata', chunk)
            self.assertLessEqual(len(chunk['content']), 20)


if __name__ == '__main__':
    unittest.main()
