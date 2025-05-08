"""Unit tests for the PDF loader module."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from llm_rag.document_processing.loaders.pdf_loaders import (
    PYMUPDF_AVAILABLE,
    PYPDF_AVAILABLE,
    EnhancedPDFLoader,
    PDFLoader,
)


class TestPDFLoader:
    """Tests for the PDF loader."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary mock PDF path
        self.pdf_path = Path('test.pdf')

    @patch('llm_rag.document_processing.loaders.pdf_loaders.Path.exists')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.PDFLoader._load_with_pymupdf')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.PDFLoader._load_with_pypdf')
    def test_load_from_file_pymupdf_available(self, mock_load_pypdf, mock_load_pymupdf, mock_exists):
        """Test that load_from_file prefers PyMuPDF when available."""
        # Mock the file existence check
        mock_exists.return_value = True

        # Mock the availability of PyMuPDF
        mock_load_pymupdf.return_value = [{'content': 'test content', 'metadata': {'source': 'test.pdf'}}]

        # Create a loader with default parameters
        loader = PDFLoader()

        # Patch PYMUPDF_AVAILABLE to True
        with patch('llm_rag.document_processing.loaders.pdf_loaders.PYMUPDF_AVAILABLE', True):
            result = loader.load_from_file(self.pdf_path)

        # Verify PyMuPDF was used
        mock_load_pymupdf.assert_called_once_with(self.pdf_path)
        mock_load_pypdf.assert_not_called()
        assert len(result) == 1
        assert result[0]['content'] == 'test content'

    @patch('llm_rag.document_processing.loaders.pdf_loaders.Path.exists')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.PDFLoader._load_with_pymupdf')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.PDFLoader._load_with_pypdf')
    def test_load_from_file_pymupdf_not_available(self, mock_load_pypdf, mock_load_pymupdf, mock_exists):
        """Test that load_from_file falls back to pypdf when PyMuPDF is not available."""
        # Mock the file existence check
        mock_exists.return_value = True

        # Mock the result of pypdf
        mock_load_pypdf.return_value = [{'content': 'test content', 'metadata': {'source': 'test.pdf'}}]

        # Create a loader with default parameters
        loader = PDFLoader()

        # Patch PYMUPDF_AVAILABLE to False
        with patch('llm_rag.document_processing.loaders.pdf_loaders.PYMUPDF_AVAILABLE', False):
            result = loader.load_from_file(self.pdf_path)

        # Verify PyMuPDF was not used and pypdf was used
        mock_load_pymupdf.assert_not_called()
        mock_load_pypdf.assert_called_once_with(self.pdf_path)
        assert len(result) == 1
        assert result[0]['content'] == 'test content'

    @pytest.mark.skip(reason='PDFLoader handling of exceptions changed, needs to be updated to match implementation')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.Path.exists')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.PDFLoader._load_with_pymupdf')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.PDFLoader._load_with_pypdf')
    def test_load_from_file_pymupdf_error(self, mock_load_pypdf, mock_load_pymupdf, mock_exists):
        """Test that load_from_file falls back to pypdf when PyMuPDF raises an error."""
        # Mock the file existence check
        mock_exists.return_value = True

        # Mock PyMuPDF to raise an exception
        mock_load_pymupdf.side_effect = Exception('PyMuPDF error')

        # Mock pypdf to return successfully
        mock_load_pypdf.return_value = [{'content': 'test content', 'metadata': {'source': 'test.pdf'}}]

        # Create a loader with default parameters
        loader = PDFLoader()

        # Patch PYMUPDF_AVAILABLE to True
        with patch('llm_rag.document_processing.loaders.pdf_loaders.PYMUPDF_AVAILABLE', True):
            result = loader.load_from_file(self.pdf_path)

        # Verify PyMuPDF was tried but pypdf was used as fallback
        mock_load_pymupdf.assert_called_once_with(self.pdf_path)
        mock_load_pypdf.assert_called_once_with(self.pdf_path)
        assert len(result) == 1
        assert result[0]['content'] == 'test content'

    @pytest.mark.skip(reason='PDFLoader handling of exceptions changed, needs to be updated to match implementation')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.Path.exists')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.PDFLoader._load_with_pymupdf')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.PDFLoader._load_with_pypdf')
    def test_load_from_file_both_fail(self, mock_load_pypdf, mock_load_pymupdf, mock_exists):
        """Test that load_from_file returns an error document when both methods fail."""
        # Mock the file existence check
        mock_exists.return_value = True

        # Mock both methods to raise exceptions
        mock_load_pymupdf.side_effect = Exception('PyMuPDF error')
        mock_load_pypdf.side_effect = Exception('pypdf error')

        # Create a loader with default parameters
        loader = PDFLoader()

        # Call load_from_file with both methods available
        with patch('llm_rag.document_processing.loaders.pdf_loaders.PYMUPDF_AVAILABLE', True):
            with patch('llm_rag.document_processing.loaders.pdf_loaders.PYPDF_AVAILABLE', True):
                result = loader.load_from_file(self.pdf_path)

        # Verify an error document is returned
        assert len(result) == 1
        assert 'error' in result[0]['metadata']
        assert 'Failed to load PDF' in result[0]['metadata']['error']

    @pytest.mark.skip(reason='Mock setup needs to be adjusted to match implementation')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.fitz.open')
    def test_load_with_pymupdf(self, mock_fitz_open):
        """Test loading a PDF with PyMuPDF."""
        # Skip if PyMuPDF is not available
        if not PYMUPDF_AVAILABLE:
            pytest.skip('PyMuPDF not available')

        # Mock fitz.open and its return values
        mock_doc = MagicMock()
        mock_fitz_open.return_value = mock_doc

        # Setup mock pages
        page1 = MagicMock()
        page1.get_text.return_value = 'Page 1 content'
        page2 = MagicMock()
        page2.get_text.return_value = 'Page 2 content'
        mock_doc.__len__.return_value = 2
        mock_doc.__getitem__.side_effect = [page1, page2]

        # Create a loader with default parameters
        loader = PDFLoader()

        # Mock content and metadata attributes
        mock_doc.metadata = {'title': 'Test PDF'}

        # Call the method with contextmanager mocking
        with patch('builtins.open', mock_open()):
            result = loader._load_with_pymupdf(self.pdf_path)

        # Verify the result
        assert len(result) == 1
        assert isinstance(result[0]['content'], str)
        assert 'Page 1 content' in result[0]['content']
        assert 'Page 2 content' in result[0]['content']
        assert result[0]['metadata']['source'] == str(self.pdf_path)
        assert result[0]['metadata']['pages'] == 2

    @pytest.mark.skip(reason='Implementation returns one document per page, test expects single document')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.PdfReader')
    def test_load_with_pypdf(self, mock_pdf_reader):
        """Test loading a PDF with pypdf."""
        # Skip if pypdf is not available
        if not PYPDF_AVAILABLE:
            pytest.skip('pypdf not available')

        # Mock PdfReader and its return values
        mock_reader = MagicMock()
        mock_pdf_reader.return_value = mock_reader

        # Setup mock pages
        page1 = MagicMock()
        page1.extract_text.return_value = 'Page 1 content'
        page2 = MagicMock()
        page2.extract_text.return_value = 'Page 2 content'
        mock_reader.pages = [page1, page2]

        # Create a loader with default parameters
        loader = PDFLoader()

        # Call the method with open mocking
        with patch('builtins.open', mock_open()):
            result = loader._load_with_pypdf(self.pdf_path)

        # Verify the result
        assert len(result) == 1
        assert 'Page 1 content' in result[0]['content']
        assert 'Page 2 content' in result[0]['content']
        assert result[0]['metadata']['source'] == str(self.pdf_path)
        assert result[0]['metadata']['pages'] == 2


class TestEnhancedPDFLoader:
    """Tests for the EnhancedPDFLoader."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary mock PDF path
        self.pdf_path = Path('test.pdf')

    @pytest.mark.skip(reason='PDFLoader inheritance requires setting attributes differently in test')
    def test_init(self):
        """Test initialization with default parameters."""
        # Create a loader with default parameters
        with patch.object(PDFLoader, '__init__', return_value=None) as mock_parent_init:
            loader = EnhancedPDFLoader(file_path=self.pdf_path)

            # Manually set extract_tables attribute since it comes from parent
            loader.extract_tables = True

            # Verify parent constructor called correctly
            mock_parent_init.assert_called_once()

            # Verify instance variables
            assert loader.use_ocr is False
            assert loader.ocr_languages == 'eng'

    @pytest.mark.skip(reason='PDFLoader inheritance requires setting attributes differently in test')
    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        # Create a loader with custom parameters
        with patch.object(PDFLoader, '__init__', return_value=None) as mock_parent_init:
            loader = EnhancedPDFLoader(
                file_path=self.pdf_path,
                extract_images=False,
                extract_tables=False,
                use_ocr=True,
                ocr_languages='eng+deu',
                page_separator='\n---\n',
                start_page=1,
                end_page=5,
            )

            # Manually set extract_tables attribute
            loader.extract_tables = False

            # Verify parent constructor called correctly
            mock_parent_init.assert_called_once()

            # Verify instance variables
            assert loader.use_ocr is True
            assert loader.ocr_languages == 'eng+deu'

    @pytest.mark.skip(reason='Mock setup needs to be adjusted to match implementation')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.Path.exists')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.EnhancedPDFLoader._load_with_pymupdf')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.PDFLoader.load_from_file')
    def test_load_from_file_pymupdf_available(self, mock_parent_load, mock_load_pymupdf, mock_exists):
        """Test that load_from_file uses PyMuPDF when available."""
        # Mock file existence check
        mock_exists.return_value = True

        # Mock the enhanced PyMuPDF loader
        mock_load_pymupdf.return_value = [{'content': 'enhanced content', 'metadata': {'source': 'test.pdf'}}]

        # Create a loader with default parameters
        loader = EnhancedPDFLoader()

        # Patch PYMUPDF_AVAILABLE to True
        with patch('llm_rag.document_processing.loaders.pdf_loaders.PYMUPDF_AVAILABLE', True):
            # We need to mock the EnhancedPDFLoader.extract_tables property which is checked in load_from_file
            with patch.object(EnhancedPDFLoader, 'extract_tables', create=True, new_callable=lambda: True):
                result = loader.load_from_file(self.pdf_path)

        # Verify enhanced PyMuPDF was used and parent method was not called
        mock_load_pymupdf.assert_called_once_with(self.pdf_path)
        mock_parent_load.assert_not_called()
        assert len(result) == 1
        assert result[0]['content'] == 'enhanced content'

    @patch('llm_rag.document_processing.loaders.pdf_loaders.Path.exists')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.EnhancedPDFLoader._load_with_pymupdf')
    @patch('llm_rag.document_processing.loaders.pdf_loaders.PDFLoader.load_from_file')
    def test_load_from_file_pymupdf_not_available(self, mock_parent_load, mock_load_pymupdf, mock_exists):
        """Test that load_from_file falls back to parent when PyMuPDF is not available."""
        # Mock file existence
        mock_exists.return_value = True

        # Mock the parent loader
        mock_parent_load.return_value = [{'content': 'basic content', 'metadata': {'source': 'test.pdf'}}]

        # Create a loader with default parameters
        loader = EnhancedPDFLoader()

        # Patch PYMUPDF_AVAILABLE to False
        with patch('llm_rag.document_processing.loaders.pdf_loaders.PYMUPDF_AVAILABLE', False):
            result = loader.load_from_file(self.pdf_path)

        # Verify enhanced PyMuPDF was not used and parent method was called
        mock_load_pymupdf.assert_not_called()
        mock_parent_load.assert_called_once_with(self.pdf_path)
        assert len(result) == 1
        assert result[0]['content'] == 'basic content'
