"""Tests for the document processors module."""

import unittest
from unittest.mock import MagicMock

from src.llm_rag.document_processing.processors import (
    DocumentProcessor,
    TextSplitter,
)


class TestTextSplitter(unittest.TestCase):
    """Tests for the TextSplitter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.text_splitter = TextSplitter(chunk_size=100, chunk_overlap=20, separators=['\n\n', '\n', ' '])

    def test_initialization(self):
        """Test that the TextSplitter initializes correctly."""
        # Access private attributes or use properties based on implementation
        self.assertEqual(self.text_splitter.splitter._chunk_size, 100)
        self.assertEqual(self.text_splitter.splitter._chunk_overlap, 20)
        self.assertEqual(self.text_splitter.splitter._separators, ['\n\n', '\n', ' '])

    def test_split_text(self):
        """Test splitting text into chunks."""
        text = 'This is a test document. ' * 10
        chunks = self.text_splitter.split_text(text)

        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1)

        # Check that each chunk is no longer than the chunk size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

    def test_split_documents(self):
        """Test splitting documents into chunks."""
        documents = [{'content': 'This is a test document. ' * 10, 'metadata': {'source': 'test1.txt'}}]

        chunks = self.text_splitter.split_documents(documents)

        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1)

        # Check that each chunk has the correct metadata
        for chunk in chunks:
            self.assertEqual(chunk['metadata']['source'], 'test1.txt')

            # Check that each chunk is no longer than the chunk size
            self.assertLessEqual(len(chunk['content']), 100)


class TestDocumentProcessor(unittest.TestCase):
    """Tests for the DocumentProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.text_splitter = MagicMock()
        self.document_processor = DocumentProcessor(self.text_splitter)

    def test_initialization(self):
        """Test that the DocumentProcessor initializes correctly."""
        self.assertEqual(self.document_processor.text_splitter, self.text_splitter)

    def test_process_empty_documents(self):
        """Test processing empty documents."""
        documents = [
            {'content': '', 'metadata': {'source': 'empty.txt'}},
            {'content': '   ', 'metadata': {'source': 'whitespace.txt'}},
            {'content': None, 'metadata': {'source': 'none.txt'}},
        ]

        # Configure the mock to return an empty list
        self.text_splitter.split_documents.return_value = []

        result = self.document_processor.process(documents)

        # Check that the result is an empty list
        self.assertEqual(result, [])

        # Check that split_documents was called with an empty list
        self.text_splitter.split_documents.assert_called_once_with([])

    def test_process_valid_documents(self):
        """Test processing valid documents."""
        documents = [
            {'content': 'Document 1', 'metadata': {'source': 'doc1.txt'}},
            {'content': 'Document 2', 'metadata': {'source': 'doc2.txt'}},
        ]

        # Configure the mock to return the same documents
        self.text_splitter.split_documents.return_value = documents

        result = self.document_processor.process(documents)

        # Check that the result is the same as the input
        self.assertEqual(result, documents)

        # Check that split_documents was called with the input documents
        self.text_splitter.split_documents.assert_called_once_with(documents)

    def test_process_mixed_documents(self):
        """Test processing a mix of valid and invalid documents."""
        documents = [
            {'content': 'Document 1', 'metadata': {'source': 'doc1.txt'}},
            {'content': '', 'metadata': {'source': 'empty.txt'}},
            {'content': 'Document 2', 'metadata': {'source': 'doc2.txt'}},
        ]

        valid_documents = [
            {'content': 'Document 1', 'metadata': {'source': 'doc1.txt'}},
            {'content': 'Document 2', 'metadata': {'source': 'doc2.txt'}},
        ]

        # Configure the mock to return the valid documents
        self.text_splitter.split_documents.return_value = valid_documents

        result = self.document_processor.process(documents)

        # Check that the result contains only the valid documents
        self.assertEqual(result, valid_documents)

        # Check that split_documents was called with the valid documents
        self.text_splitter.split_documents.assert_called_once_with(valid_documents)


if __name__ == '__main__':
    unittest.main()
