"""Unit tests for the JSONLoader class."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_rag.document_processing.loaders.json_loader import JSONLoader


class TestJSONLoader(unittest.TestCase):
    """Test suite for the JSONLoader class."""

    def setUp(self):
        """Set up test fixtures before each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        # Create a sample JSON file
        self.json_data = {
            'title': 'Test Document',
            'content': 'This is a test document.',
            'metadata': {'author': 'Test Author', 'date': '2023-01-01'},
        }

        self.json_path = Path(self.temp_dir.name) / 'test.json'
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.json_data, f)

        # Create a sample JSON Lines file
        self.jsonl_path = Path(self.temp_dir.name) / 'test.jsonl'
        with open(self.jsonl_path, 'w', encoding='utf-8') as f:
            for i in range(3):
                f.write(json.dumps({'id': i, 'content': f'Entry {i} content', 'tags': ['test', f'tag{i}']}) + '\n')

    def test_initialization(self):
        """Test initialization of JSONLoader."""
        # Default initialization
        loader = JSONLoader()
        self.assertIsNone(loader.file_path)
        self.assertIsNone(loader.content_key)
        self.assertEqual(loader.metadata_keys, [])
        self.assertIsNone(loader.jq_filter)
        self.assertFalse(loader.json_lines)

        # With parameters
        loader = JSONLoader(
            file_path=self.json_path, content_key='content', metadata_keys=['title', 'metadata.author'], json_lines=True
        )
        self.assertEqual(loader.file_path, Path(self.json_path))
        self.assertEqual(loader.content_key, 'content')
        self.assertEqual(loader.metadata_keys, ['title', 'metadata.author'])
        self.assertTrue(loader.json_lines)

    def test_load_without_file_path(self):
        """Test load method raises ValueError when file_path is not provided."""
        loader = JSONLoader()
        with self.assertRaises(ValueError):
            loader.load()

    def test_load_from_file_not_found(self):
        """Test load_from_file raises FileNotFoundError for non-existent file."""
        loader = JSONLoader()
        with self.assertRaises(FileNotFoundError):
            loader.load_from_file('non_existent_file.json')

    def test_load_regular_json(self):
        """Test loading a regular JSON file."""
        loader = JSONLoader(self.json_path)
        documents = loader.load()

        self.assertEqual(len(documents), 1)
        doc = documents[0]
        self.assertIn('content', doc)
        self.assertIn('metadata', doc)
        self.assertEqual(doc['metadata']['source'], str(self.json_path))

        # The content should be the entire JSON as a string
        self.assertIn('Test Document', doc['content'])
        self.assertIn('This is a test document', doc['content'])

    def test_load_with_content_key(self):
        """Test loading JSON with specific content key."""
        loader = JSONLoader(file_path=self.json_path, content_key='content')
        documents = loader.load()

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]['content'], 'This is a test document.')

    def test_load_with_metadata_keys(self):
        """Test loading JSON with specific metadata keys."""
        loader = JSONLoader(file_path=self.json_path, content_key='content', metadata_keys=['title'])
        documents = loader.load()

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]['content'], 'This is a test document.')
        self.assertEqual(documents[0]['metadata']['title'], 'Test Document')

    def test_load_json_lines(self):
        """Test loading a JSON Lines file."""
        loader = JSONLoader(
            file_path=self.jsonl_path, json_lines=True, content_key='content', metadata_keys=['id', 'tags']
        )
        documents = loader.load()

        self.assertEqual(len(documents), 3)

        # Check the first document
        self.assertEqual(documents[0]['content'], 'Entry 0 content')
        self.assertEqual(documents[0]['metadata']['id'], 0)
        self.assertEqual(documents[0]['metadata']['line'], 1)
        self.assertEqual(documents[0]['metadata']['tags'], ['test', 'tag0'])

        # Check the last document
        self.assertEqual(documents[2]['content'], 'Entry 2 content')
        self.assertEqual(documents[2]['metadata']['id'], 2)
        self.assertEqual(documents[2]['metadata']['line'], 3)
        self.assertEqual(documents[2]['metadata']['tags'], ['test', 'tag2'])

    @patch('llm_rag.document_processing.loaders.json_loader.logger')
    def test_load_invalid_json_lines(self, mock_logger):
        """Test handling invalid lines in JSON Lines file."""
        # Create a JSON Lines file with one invalid line
        bad_jsonl_path = Path(self.temp_dir.name) / 'bad.jsonl'
        with open(bad_jsonl_path, 'w', encoding='utf-8') as f:
            f.write('{"id": 1, "content": "Valid line"}\n')
            f.write('invalid json\n')
            f.write('{"id": 2, "content": "Another valid line"}\n')

        loader = JSONLoader(file_path=bad_jsonl_path, json_lines=True, content_key='content')
        documents = loader.load()

        # Should get 2 valid documents despite 1 invalid line
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]['content'], 'Valid line')
        self.assertEqual(documents[1]['content'], 'Another valid line')

        # Check that warning was logged
        mock_logger.warning.assert_called_once()
        self.assertIn('Error decoding JSON at line 2', mock_logger.warning.call_args[0][0])

    @pytest.mark.skipif(not os.environ.get('JQ_AVAILABLE'), reason='jq library not available')
    def test_with_jq_filter(self):
        """Test using jq filter with JSON."""
        # This test is skipped if jq is not available
        loader = JSONLoader(file_path=self.json_path, jq_filter='.metadata', content_key='author')

        # Mock the jq import and filter
        with patch('llm_rag.document_processing.loaders.json_loader.jq') as mock_jq:
            mock_compile = MagicMock()
            mock_input = MagicMock()
            mock_all = MagicMock()

            mock_all.return_value = {'author': 'Test Author', 'date': '2023-01-01'}
            mock_input.return_value = mock_all
            mock_compile.return_value = mock_input
            mock_jq.compile.return_value = mock_compile

            # Set has_jq flag to True
            loader._has_jq = True

            documents = loader.load()

            # Check that jq was called correctly
            mock_jq.compile.assert_called_once_with('.metadata')

            # Check the results
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]['content'], 'Test Author')


if __name__ == '__main__':
    unittest.main()
