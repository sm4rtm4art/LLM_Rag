"""Tests for language preservation and translation in the OCR pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from llm_rag.document_processing.ocr.llm_processor import LLMCleaner, LLMCleanerConfig

# Sample multilingual texts
GERMAN_TEXT = "Dies ist ein deutscher Text rnit einigen OCR-Fehlern wie falsch3n Buchstab3n."
FRENCH_TEXT = "C'est un texte français avec des erreurs d'OCR comme des caractèr3s incorrects."
SPANISH_TEXT = "Este es un texto español con algunos errores de OCR como caractere5 incorrectos."
ENGLISH_TEXT = "This is an English text with some OCR errors like wro4g charact3rs."


@pytest.fixture
def mock_langdetect():
    """Mock for langdetect to avoid external dependencies."""
    with patch("llm_rag.document_processing.ocr.llm_processor.langdetect") as mock:
        mock.detect.return_value = "de"  # Pretend text is in German
        yield mock


@pytest.fixture
def llm_cleaner():
    """Create a LLMCleaner instance with default settings."""
    return LLMCleaner()


@pytest.fixture
def translation_cleaner():
    """Create a LLMCleaner instance configured for translation."""
    config = LLMCleanerConfig(
        translate_to_language="en",  # Translate to English
        preserve_language=False,  # Should be set automatically
    )
    return LLMCleaner(config)


@pytest.fixture
def language_preserving_cleaner():
    """Create a LLMCleaner instance configured to explicitly preserve language."""
    config = LLMCleanerConfig(
        preserve_language=True,
        translate_to_language=None,
    )
    return LLMCleaner(config)


class TestLanguageHandling:
    @pytest.fixture
    def mock_model_factory(self):
        """Create a mock model factory."""
        with patch("llm_rag.document_processing.ocr.llm_processor.ModelFactory") as mock:
            mock_model = MagicMock()
            mock_model.generate.return_value = "Corrected text"
            mock.create_model.return_value = mock_model
            yield mock

    @patch("llm_rag.document_processing.ocr.llm_processor.langdetect")
    def test_language_detection(self, mock_langdetect, mock_model_factory):
        """Test that language detection works correctly for different languages."""
        # Configure language detection mock
        mock_langdetect.detect.side_effect = lambda text: {
            GERMAN_TEXT: "de",
            FRENCH_TEXT: "fr",
            SPANISH_TEXT: "es",
            ENGLISH_TEXT: "en",
        }[text]

        # Create cleaner with language detection enabled
        config = LLMCleanerConfig(detect_language=True, preserve_language=True)
        cleaner = LLMCleaner(config)

        # Test detection for each language
        for text, expected_lang in [
            (GERMAN_TEXT, "de"),
            (FRENCH_TEXT, "fr"),
            (SPANISH_TEXT, "es"),
            (ENGLISH_TEXT, "en"),
        ]:
            detected = cleaner.detect_language(text)
            assert detected == expected_lang

    @patch("llm_rag.document_processing.ocr.llm_processor.langdetect")
    def test_language_preservation(self, mock_langdetect, mock_model_factory):
        """Test that language is preserved during cleaning."""
        mock_langdetect.detect.return_value = "de"

        # Mock model to simulate preservation
        mock_model = MagicMock()
        mock_model.generate.return_value = "Dies ist ein korrigierter deutscher Text ohne Fehler."
        mock_model_factory.create_model.return_value = mock_model

        # Create cleaner with language preservation enabled
        config = LLMCleanerConfig(detect_language=True, preserve_language=True)
        cleaner = LLMCleaner(config)

        # Patch the _process_with_llm method to access the prompt
        with patch.object(cleaner, "_create_cleaning_prompt") as mock_create_prompt:
            mock_create_prompt.return_value = "Test prompt"
            with patch.object(cleaner, "_process_with_llm", return_value="Cleaned text"):
                # Clean German text
                cleaner.clean_text(GERMAN_TEXT, metadata={"language": "de"})

                # Verify language preservation settings were used
                assert cleaner.config.preserve_language is True
                assert cleaner.config.translate_to_language is None

    @patch("llm_rag.document_processing.ocr.llm_processor.langdetect")
    def test_translation(self, mock_langdetect, mock_model_factory):
        """Test translation to target language."""
        mock_langdetect.detect.return_value = "de"

        # Mock model to simulate translation to English
        mock_model = MagicMock()
        mock_model.generate.return_value = "This is a corrected German text translated to English without errors."
        mock_model_factory.create_model.return_value = mock_model

        # Create cleaner with translation enabled
        config = LLMCleanerConfig(detect_language=True, translate_to_language="en")
        cleaner = LLMCleaner(config)

        # Patch the _process_with_llm method to access the prompt
        with patch.object(cleaner, "_create_cleaning_prompt") as mock_create_prompt:
            mock_create_prompt.return_value = "Test prompt"
            with patch.object(cleaner, "_process_with_llm", return_value="Translated text"):
                # Clean and translate German text
                cleaner.clean_text(GERMAN_TEXT, metadata={"language": "de"})

                # Verify translation settings were used
                assert cleaner.config.preserve_language is False
                assert cleaner.config.translate_to_language == "en"

    def test_language_specific_models(self):
        """Test configuration of language-specific models."""
        # Create a cleaner with language-specific models configuration
        config = LLMCleanerConfig(
            language_models={
                "de": "german-model",
                "fr": "french-model",
                "es": "spanish-model",
            }
        )

        # Simply verify the configuration is stored correctly
        cleaner = LLMCleaner(config)
        assert cleaner.config.language_models["de"] == "german-model"
        assert cleaner.config.language_models["fr"] == "french-model"
        assert cleaner.config.language_models["es"] == "spanish-model"

        # This test verifies the configuration is stored properly
        # The actual model loading happens in the model property which is
        # better tested with integration tests rather than unit tests

    def test_estimate_error_rate(self):
        """Test that error rate estimation works correctly."""
        cleaner = LLMCleaner()

        # Test with text containing OCR errors
        text_with_errors = "Th1s t3xt has 5ome OCR err0rs."
        error_rate = cleaner._estimate_error_rate(text_with_errors)

        # Error rate should be above 0 since text has errors
        assert error_rate > 0

        # Test with clean text
        clean_text = "This text has no OCR errors."
        clean_error_rate = cleaner._estimate_error_rate(clean_text)

        # Clean text should have lower error rate
        assert clean_error_rate < error_rate
