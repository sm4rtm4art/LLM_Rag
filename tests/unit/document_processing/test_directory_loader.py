"""Unit tests for the DirectoryLoader.

This module contains tests for the DirectoryLoader implementation.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from llm_rag.document_processing.loaders import DirectoryLoader


class TestDirectoryLoader:
    """Test cases for the DirectoryLoader class."""

    def test_initialization(self):
        """Test initialization of DirectoryLoader."""
        # Create the loader with default parameters
        loader = DirectoryLoader(directory_path='test_dir')

        assert loader.directory_path == Path('test_dir')
        assert loader.glob_pattern == '*.*'
        assert loader.recursive is False
        assert loader.exclude_patterns == []
        assert loader.exclude_hidden is True
        assert loader.loader_kwargs == {}

        # Create with custom parameters
        loader = DirectoryLoader(
            directory_path='test_dir',
            glob_pattern='*.txt',
            recursive=True,
            exclude_patterns=['*.temp', '*.log'],
            exclude_hidden=False,
            loader_kwargs={'encoding': 'utf-8'},
        )

        assert loader.directory_path == Path('test_dir')
        assert loader.glob_pattern == '*.txt'
        assert loader.recursive is True
        assert loader.exclude_patterns == ['*.temp', '*.log']
        assert loader.exclude_hidden is False
        assert loader.loader_kwargs == {'encoding': 'utf-8'}

    def test_load_method_calls_load_from_directory(self):
        """Test that load() calls load_from_directory with the correct parameters."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.is_dir', return_value=True):
            loader = DirectoryLoader(
                directory_path='test_dir',
                glob_pattern='*.txt',
                recursive=True,
                exclude_patterns=['*.temp'],
                exclude_hidden=False,
                loader_kwargs={'encoding': 'utf-8'},
            )

            # Mock the load_from_directory method
            with patch.object(DirectoryLoader, 'load_from_directory') as mock_load_from_directory:
                mock_load_from_directory.return_value = [{'content': 'Test', 'metadata': {}}]

                # Act
                result = loader.load()

                # Assert
                mock_load_from_directory.assert_called_once_with(
                    Path('test_dir'), '*.txt', True, ['*.temp'], False, {'encoding': 'utf-8'}
                )
                assert result == [{'content': 'Test', 'metadata': {}}]

    def test_load_directory_not_found(self):
        """Test load() when directory doesn't exist."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=False):
            loader = DirectoryLoader(directory_path='nonexistent')

            # Act & Assert
            with pytest.raises(NotADirectoryError, match='Directory not found'):
                loader.load()

    def test_load_from_directory_directory_not_found(self):
        """Test load_from_directory() when directory doesn't exist."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=False):
            # Act & Assert
            with pytest.raises(NotADirectoryError):
                DirectoryLoader.load_from_directory('nonexistent')

    def test_load_from_directory_with_files(self):
        """Test load_from_directory() with multiple files."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.is_dir', return_value=True):
            # Mock glob to return some files
            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = [Path('file1.txt'), Path('file2.txt'), Path('.hidden.txt')]

                # Mock is_file to return True for all files
                with patch('pathlib.Path.is_file', return_value=True):
                    # Mock load_document
                    with patch(
                        'llm_rag.document_processing.loaders.directory_loader.load_document'
                    ) as mock_load_document:
                        mock_load_document.side_effect = [
                            [{'content': 'Content 1', 'metadata': {}}],
                            [{'content': 'Content 2', 'metadata': {}}],
                        ]

                        # Act
                        result = DirectoryLoader.load_from_directory('test_dir')

                        # Assert
                        assert len(result) == 2
                        assert result[0]['content'] == 'Content 1'
                        assert result[1]['content'] == 'Content 2'
                        assert mock_load_document.call_count == 2

    def test_load_from_directory_with_exclude_patterns(self):
        """Test load_from_directory() with exclude patterns."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.is_dir', return_value=True):
            # Mock glob to return some files
            with patch('pathlib.Path.glob') as mock_glob:
                # For the main glob
                mock_glob.side_effect = lambda pattern: (
                    [Path('file1.txt'), Path('file2.txt'), Path('temp.txt')]
                    if pattern == '*.*'
                    else [Path('temp.txt')]  # For the exclude pattern
                )

                # Mock is_file to return True for all files
                with patch('pathlib.Path.is_file', return_value=True):
                    # Mock load_document
                    with patch(
                        'llm_rag.document_processing.loaders.directory_loader.load_document'
                    ) as mock_load_document:
                        mock_load_document.side_effect = [
                            [{'content': 'Content 1', 'metadata': {}}],
                            [{'content': 'Content 2', 'metadata': {}}],
                        ]

                        # Act
                        result = DirectoryLoader.load_from_directory('test_dir', exclude_patterns=['temp.txt'])

                        # Assert
                        assert len(result) == 2
                        assert result[0]['content'] == 'Content 1'
                        assert result[1]['content'] == 'Content 2'
                        assert mock_load_document.call_count == 2

    def test_load_from_directory_with_error(self):
        """Test load_from_directory() when a file loader raises an error."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.is_dir', return_value=True):
            # Mock glob to return some files
            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = [Path('file1.txt'), Path('file2.txt')]

                # Mock is_file to return True for all files
                with patch('pathlib.Path.is_file', return_value=True):
                    # Mock load_document
                    with patch(
                        'llm_rag.document_processing.loaders.directory_loader.load_document'
                    ) as mock_load_document:
                        mock_load_document.side_effect = [
                            [{'content': 'Content 1', 'metadata': {}}],
                            Exception('Test error'),
                        ]

                        # Act
                        result = DirectoryLoader.load_from_directory('test_dir')

                        # Assert
                        assert len(result) == 1
                        assert result[0]['content'] == 'Content 1'
                        assert mock_load_document.call_count == 2
