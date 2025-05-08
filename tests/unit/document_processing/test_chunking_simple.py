"""Simple tests for the chunking module."""

import unittest

from src.llm_rag.document_processing.chunking import (
    CharacterTextChunker,
    RecursiveTextChunker,
)


class TestChunkersValidation(unittest.TestCase):
    """Tests for the validation logic in chunkers."""

    def test_character_chunker_validation(self):
        """Test validation in CharacterTextChunker."""
        # Valid initialization should not raise exceptions
        chunker = CharacterTextChunker(chunk_size=100, chunk_overlap=20, separator='\n')
        self.assertEqual(chunker.chunk_size, 100)
        self.assertEqual(chunker.chunk_overlap, 20)
        self.assertEqual(chunker.separator, '\n')

        # Test invalid chunk_size
        with self.assertRaises(ValueError):
            CharacterTextChunker(chunk_size=0)

        with self.assertRaises(ValueError):
            CharacterTextChunker(chunk_size=-10)

        # Test invalid chunk_overlap (negative)
        with self.assertRaises(ValueError):
            CharacterTextChunker(chunk_overlap=-1)

        # Test invalid chunk_overlap (>= chunk_size)
        with self.assertRaises(ValueError):
            CharacterTextChunker(chunk_size=100, chunk_overlap=100)

        with self.assertRaises(ValueError):
            CharacterTextChunker(chunk_size=100, chunk_overlap=150)

    def test_recursive_chunker_validation(self):
        """Test validation in RecursiveTextChunker."""
        # Valid initialization should not raise exceptions
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20, separators=['\n\n', '\n', '. ', ' '])
        self.assertEqual(chunker.chunk_size, 100)
        self.assertEqual(chunker.chunk_overlap, 20)
        self.assertEqual(chunker.separators, ['\n\n', '\n', '. ', ' '])

        # Test default separators
        chunker = RecursiveTextChunker(chunk_size=100, chunk_overlap=20)
        self.assertEqual(chunker.chunk_size, 100)
        self.assertEqual(chunker.chunk_overlap, 20)
        self.assertEqual(chunker.separators, ['\n\n', '\n', '. ', '.', ' ', ''])

        # Test invalid chunk_size
        with self.assertRaises(ValueError):
            RecursiveTextChunker(chunk_size=0)

        with self.assertRaises(ValueError):
            RecursiveTextChunker(chunk_size=-10)

        # Test invalid chunk_overlap (negative)
        with self.assertRaises(ValueError):
            RecursiveTextChunker(chunk_overlap=-1)

        # Test invalid chunk_overlap (>= chunk_size)
        with self.assertRaises(ValueError):
            RecursiveTextChunker(chunk_size=100, chunk_overlap=100)

        with self.assertRaises(ValueError):
            RecursiveTextChunker(chunk_size=100, chunk_overlap=150)


if __name__ == '__main__':
    unittest.main()
