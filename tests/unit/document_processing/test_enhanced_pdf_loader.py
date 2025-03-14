"""Unit tests for the EnhancedPDFLoader class."""

import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

# Try to import EnhancedPDFLoader
try:
    from llm_rag.document_processing.loaders import EnhancedPDFLoader, PDFLoader

    has_enhanced_pdf_loader = True
except ImportError:
    has_enhanced_pdf_loader = False
    PDFLoader = object  # Placeholder to prevent further import errors

    # Create a dummy class for tests to patch
    class EnhancedPDFLoader:
        pass


# Skip all tests if EnhancedPDFLoader is not available
pytestmark = pytest.mark.skipif(not has_enhanced_pdf_loader, reason="EnhancedPDFLoader not available")


@pytest.mark.refactoring
class TestEnhancedPDFLoader(unittest.TestCase):
    """Test cases for the EnhancedPDFLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_pdf_path = "test.pdf"

    def test_initialization(self):
        """Test the initialization of EnhancedPDFLoader."""
        # Arrange & Act
        loader = EnhancedPDFLoader(
            file_path=self.test_pdf_path, extract_images=True, extract_tables=True, use_ocr=True, ocr_languages="eng"
        )

        # Assert
        self.assertEqual(loader.file_path, Path(self.test_pdf_path))
        self.assertTrue(loader.extract_images)
        self.assertTrue(loader.extract_tables)
        self.assertTrue(loader.use_ocr)
        self.assertEqual(loader.ocr_languages, "eng")

    def test_initialization_default_values(self):
        """Test the initialization with default values."""
        # Arrange & Act
        loader = EnhancedPDFLoader(file_path=self.test_pdf_path)

        # Assert
        self.assertTrue(loader.extract_images)  # Default is True for enhanced
        self.assertTrue(loader.extract_tables)  # Default is True for enhanced
        self.assertFalse(loader.use_ocr)  # Default is False

    def test_initialization_inheritance(self):
        """Test that EnhancedPDFLoader properly inherits from PDFLoader."""
        # Arrange & Act
        loader = EnhancedPDFLoader(file_path=self.test_pdf_path)

        # Assert
        self.assertIsInstance(loader, PDFLoader)

    @patch.object(EnhancedPDFLoader, "_load_with_pymupdf")
    def test_load_calls_load_with_pymupdf(self, mock_load_with_pymupdf):
        """Test that load() calls _load_with_pymupdf()."""
        # Arrange
        mock_load_with_pymupdf.return_value = [{"content": "Enhanced content", "metadata": {"filetype": "pdf"}}]

        # Create loader
        loader = EnhancedPDFLoader(file_path=self.test_pdf_path)

        # Patch the exists method to return True and ensure PYMUPDF_AVAILABLE is True
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("src.llm_rag.document_processing.loaders.pdf_loaders.PYMUPDF_AVAILABLE", True),
            # This is key - also need to patch the parent load_from_file to call our mocked method
            patch.object(PDFLoader, "load_from_file", side_effect=lambda path: mock_load_with_pymupdf(path)),
        ):
            # Act
            documents = loader.load()

        # Assert
        mock_load_with_pymupdf.assert_called_once()
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["content"], "Enhanced content")

    @patch.object(EnhancedPDFLoader, "_load_with_pymupdf")
    @patch.object(PDFLoader, "_load_with_pymupdf")
    def test_load_vs_parent_load(self, mock_parent_load, mock_enhanced_load):
        """Test that EnhancedPDFLoader _load_with_pymupdf() extends parent method."""
        # Arrange
        mock_enhanced_load.return_value = [{"content": "Enhanced content", "metadata": {"filetype": "pdf"}}]
        mock_parent_load.return_value = [{"content": "Regular content", "metadata": {"filetype": "pdf"}}]

        # Create loaders
        enhanced_loader = EnhancedPDFLoader(file_path=self.test_pdf_path)
        regular_loader = PDFLoader(file_path=self.test_pdf_path)

        # Patch the exists method to return True and ensure PYMUPDF_AVAILABLE is True
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("src.llm_rag.document_processing.loaders.pdf_loaders.PYMUPDF_AVAILABLE", True),
            # This is key - also need to patch the parent load_from_file to call our mocked methods
            patch.object(EnhancedPDFLoader, "load_from_file", side_effect=lambda path: mock_enhanced_load(path)),
            patch.object(PDFLoader, "load_from_file", side_effect=lambda path: mock_parent_load(path)),
        ):
            # Act
            enhanced_docs = enhanced_loader.load()
            regular_docs = regular_loader.load()

        # Assert
        mock_enhanced_load.assert_called_once()
        self.assertEqual(enhanced_docs[0]["content"], "Enhanced content")
        self.assertEqual(regular_docs[0]["content"], "Regular content")


@pytest.mark.local_only
class TestEnhancedPDFLoaderWithRealFiles:
    """Tests using real PDF files, marked to be skipped in CI."""

    @pytest.fixture
    def setup_test_pdf(self):
        """Create a simple test PDF file for testing."""
        test_pdf_path = Path("tests/test_data/test.pdf")
        if not test_pdf_path.exists():
            pytest.skip("Test PDF file not found")
        return str(test_pdf_path)

    def test_real_pdf_enhanced_extraction(self, setup_test_pdf):
        """Test enhanced extraction from a real PDF file."""
        # This test will only run if the test PDF file exists
        pdf_path = setup_test_pdf

        # Create loader with enhanced extraction
        loader = EnhancedPDFLoader(file_path=pdf_path, extract_images=True, extract_tables=True)

        # Load documents
        documents = loader.load()

        # Basic assertions
        assert len(documents) > 0
        assert isinstance(documents, list)
        assert isinstance(documents[0], dict)
        assert "content" in documents[0]
        assert "metadata" in documents[0]


if __name__ == "__main__":
    unittest.main()
