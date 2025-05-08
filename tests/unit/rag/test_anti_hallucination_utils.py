"""Unit tests for the anti_hallucination.utils module."""

import json
from unittest.mock import MagicMock, mock_open, patch

# Import only the specific function we want to test
from llm_rag.rag.anti_hallucination.utils import load_stopwords


class TestStopwords:
    """Tests for stopwords functionality."""

    def test_load_stopwords_default(self):
        """Test loading default English stopwords."""
        # Get stopwords
        stopwords = load_stopwords()

        # Verify it's a set with expected common English stopwords
        assert isinstance(stopwords, set)
        assert 'the' in stopwords
        assert 'and' in stopwords
        assert 'or' in stopwords
        assert len(stopwords) > 50  # Should have many stopwords

    def test_load_stopwords_german(self):
        """Test loading German stopwords."""
        # Get German stopwords
        stopwords = load_stopwords(language='de')

        # Verify it's a set with expected German stopwords
        assert isinstance(stopwords, set)
        assert 'und' in stopwords
        assert 'der' in stopwords
        assert 'die' in stopwords
        assert len(stopwords) > 50  # Should have many stopwords

    def test_load_stopwords_from_file(self):
        """Test loading stopwords from a file."""
        # Create a mock file with proper content
        # The function expects a simple list in the JSON file
        mock_stopwords = ['word1', 'word2', 'word3']

        with patch('builtins.open', mock_open()):
            with patch('json.load') as mock_json_load:
                # The function expects json.load to return a list
                mock_json_load.return_value = mock_stopwords

                # Call the function with test language
                stopwords = load_stopwords(language='test')

                # Verify the result contains the words from the mock data
                assert isinstance(stopwords, set)
                assert len(stopwords) == 3
                assert 'word1' in stopwords
                assert 'word2' in stopwords
                assert 'word3' in stopwords

    def test_load_stopwords_fallback_to_defaults(self, monkeypatch):
        """Test falling back to default stopwords when file not found."""
        # Mock the open function to raise a FileNotFoundError
        monkeypatch.setattr('builtins.open', mock_open())
        monkeypatch.setattr('builtins.open', MagicMock(side_effect=FileNotFoundError()))

        # Use direct patching instead of monkeypatch for the module attribute
        default_stopwords = {'en': {'default', 'words'}}
        with patch('llm_rag.rag.anti_hallucination.utils._DEFAULT_STOPWORDS', default_stopwords):
            # Call the function
            stopwords = load_stopwords()

            # Verify the result
            assert isinstance(stopwords, set)
            assert 'default' in stopwords
            assert 'words' in stopwords
            assert len(stopwords) == 2

    def test_load_stopwords_json_error(self, monkeypatch):
        """Test handling of JSON decode errors."""
        # Mock the open function and json.load
        mock_open_obj = mock_open()
        monkeypatch.setattr('builtins.open', mock_open_obj)
        monkeypatch.setattr('json.load', MagicMock(side_effect=json.JSONDecodeError('', '', 0)))

        # Use direct patching for the module attribute
        spanish_stopwords = {'es': {'palabras', 'vacías'}}
        with patch('llm_rag.rag.anti_hallucination.utils._DEFAULT_STOPWORDS', spanish_stopwords):
            # Call the function
            stopwords = load_stopwords(language='es')

            # Verify the result
            assert isinstance(stopwords, set)
            assert 'palabras' in stopwords
            assert 'vacías' in stopwords
            assert len(stopwords) == 2

    def test_load_stopwords_unknown_language(self, monkeypatch):
        """Test handling of unknown languages."""
        # Mock the open function to raise a FileNotFoundError
        monkeypatch.setattr('builtins.open', mock_open())
        monkeypatch.setattr('builtins.open', MagicMock(side_effect=FileNotFoundError()))

        # Use direct patching for the module attribute with an empty dict
        with patch('llm_rag.rag.anti_hallucination.utils._DEFAULT_STOPWORDS', {}):
            # Call the function with an unknown language
            stopwords = load_stopwords(language='unknown')

            # Verify the result is an empty set
            assert isinstance(stopwords, set)
            assert len(stopwords) == 0
