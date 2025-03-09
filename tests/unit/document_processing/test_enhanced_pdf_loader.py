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


@pytest.mark.skipif(not has_enhanced_pdf_loader, reason="EnhancedPDFLoader not available")
class TestEnhancedPDFLoader(unittest.TestCase):
    """Test cases for the EnhancedPDFLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_pdf_path = "test.pdf"
        self.output_dir = "test_output"

    def test_initialization(self):
        """Test the initialization of EnhancedPDFLoader."""
        # Arrange & Act
        loader = EnhancedPDFLoader(
            file_path=self.test_pdf_path, extract_images=True, extract_tables=True, output_dir=self.output_dir
        )

        # Assert
        self.assertEqual(loader.file_path_str, self.test_pdf_path)
        self.assertTrue(loader.extract_images)
        self.assertTrue(loader.extract_tables)
        self.assertTrue(loader.use_enhanced_extraction)
        self.assertEqual(loader.output_dir, self.output_dir)

    def test_initialization_default_values(self):
        """Test the initialization with default values."""
        # Arrange & Act
        loader = EnhancedPDFLoader(file_path=self.test_pdf_path)

        # Assert
        self.assertTrue(loader.extract_images)  # Default is True for enhanced
        self.assertTrue(loader.extract_tables)  # Default is True for enhanced
        self.assertTrue(loader.use_enhanced_extraction)
        self.assertIsNone(loader.output_dir)

    def test_initialization_inheritance(self):
        """Test that EnhancedPDFLoader properly inherits from PDFLoader."""
        # Arrange & Act
        loader = EnhancedPDFLoader(file_path=self.test_pdf_path)

        # Assert
        self.assertIsInstance(loader, PDFLoader)

    @patch.object(PDFLoader, "_load_enhanced")
    def test_load_calls_load_enhanced(self, mock_load_enhanced):
        """Test that load() calls _load_enhanced()."""
        # Arrange
        mock_load_enhanced.return_value = [{"content": "Enhanced content", "metadata": {"filetype": "pdf"}}]

        # Create loader
        loader = EnhancedPDFLoader(file_path=self.test_pdf_path)

        # Act
        documents = loader.load()

        # Assert
        mock_load_enhanced.assert_called_once()
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["content"], "Enhanced content")

    @patch.object(EnhancedPDFLoader, "_load_enhanced")
    @patch.object(PDFLoader, "load")
    def test_load_vs_parent_load(self, mock_parent_load, mock_load_enhanced):
        """Test that EnhancedPDFLoader load() differs from parent load()."""
        # Arrange
        mock_load_enhanced.return_value = [{"content": "Enhanced content", "metadata": {"filetype": "pdf_enhanced"}}]
        mock_parent_load.return_value = [{"content": "Regular content", "metadata": {"filetype": "pdf"}}]

        # Create loaders
        enhanced_loader = EnhancedPDFLoader(file_path=self.test_pdf_path)
        regular_loader = PDFLoader(file_path=self.test_pdf_path, use_enhanced_extraction=False)

        # Act
        enhanced_docs = enhanced_loader.load()
        regular_docs = regular_loader.load()

        # Assert
        mock_load_enhanced.assert_called_once()
        mock_parent_load.assert_called_once()
        self.assertEqual(enhanced_docs[0]["content"], "Enhanced content")
        self.assertEqual(enhanced_docs[0]["metadata"]["filetype"], "pdf_enhanced")
        self.assertEqual(regular_docs[0]["content"], "Regular content")
        self.assertEqual(regular_docs[0]["metadata"]["filetype"], "pdf")


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
        output_dir = "tests/test_data/output_enhanced"

        # Create the output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        # Create loader with enhanced extraction
        loader = EnhancedPDFLoader(file_path=pdf_path, output_dir=output_dir)

        # Load documents
        documents = loader.load()

        # Basic assertions
        assert len(documents) > 0
        assert isinstance(documents, list)
        assert isinstance(documents[0], dict)
        assert "content" in documents[0]
        assert "metadata" in documents[0]

        # Clean up output directory after test
        # This is commented out to allow inspection of the output
        # import shutil
        # shutil.rmtree(output_dir)


if __name__ == "__main__":
    unittest.main()
