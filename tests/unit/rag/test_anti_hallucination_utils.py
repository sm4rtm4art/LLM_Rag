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
        assert "the" in stopwords
        assert "and" in stopwords
        assert "or" in stopwords
        assert len(stopwords) > 50  # Should have many stopwords

    def test_load_stopwords_german(self):
        """Test loading German stopwords."""
        # Get German stopwords
        stopwords = load_stopwords(language="de")

        # Verify it's a set with expected common German stopwords
        assert isinstance(stopwords, set)
        assert "die" in stopwords
        assert "der" in stopwords
        assert "und" in stopwords
        assert len(stopwords) > 50  # Should have many stopwords

    @patch("os.path.join", return_value="/mock/path/stopwords_en.json")
    @patch("builtins.open", new_callable=mock_open, read_data='["test", "words"]')
    def test_load_stopwords_from_file(self, mock_file, mock_join):
        """Test loading stopwords from a file."""
        # Call the function
        result = load_stopwords()

        # Check that file was opened and results reflect mock data
        mock_file.assert_called_once_with("/mock/path/stopwords_en.json", "r", encoding="utf-8")
        assert result == {"test", "words"}

    # Use monkeypatch instead of patch for internal variables
    def test_load_stopwords_fallback_to_defaults(self, monkeypatch):
        """Test falling back to default stopwords when file not found."""
        # Mock the open function to raise a FileNotFoundError
        monkeypatch.setattr("builtins.open", mock_open())
        monkeypatch.setattr("builtins.open", MagicMock(side_effect=FileNotFoundError()))

        # Mock the _DEFAULT_STOPWORDS
        default_stopwords = {"en": {"default", "words"}}
        monkeypatch.setattr("llm_rag.rag.anti_hallucination.utils._DEFAULT_STOPWORDS", default_stopwords)

        # Call the function
        result = load_stopwords()

        # Check results are from default dict
        assert result == {"default", "words"}

    # Use monkeypatch instead of patch for internal variables
    def test_load_stopwords_json_error(self, monkeypatch):
        """Test handling of JSON decode errors."""
        # Mock the open function to raise a JSONDecodeError
        mock_open_obj = mock_open()
        monkeypatch.setattr("builtins.open", mock_open_obj)
        monkeypatch.setattr("json.load", MagicMock(side_effect=json.JSONDecodeError("", "", 0)))

        # Mock the _DEFAULT_STOPWORDS
        default_stopwords = {"es": {"palabras", "vacías"}}
        monkeypatch.setattr("llm_rag.rag.anti_hallucination.utils._DEFAULT_STOPWORDS", default_stopwords)

        # Call the function
        result = load_stopwords(language="es")

        # Check results are from default dict
        assert result == {"palabras", "vacías"}

    # Use monkeypatch instead of patch for internal variables
    def test_load_stopwords_unknown_language(self, monkeypatch):
        """Test handling of unknown languages."""
        # Mock the open function to raise a FileNotFoundError
        monkeypatch.setattr("builtins.open", mock_open())
        monkeypatch.setattr("builtins.open", MagicMock(side_effect=FileNotFoundError()))

        # Mock the _DEFAULT_STOPWORDS with an empty dict
        monkeypatch.setattr("llm_rag.rag.anti_hallucination.utils._DEFAULT_STOPWORDS", {})

        # Call the function
        result = load_stopwords(language="xx")

        # Check empty set is returned
        assert result == set()
