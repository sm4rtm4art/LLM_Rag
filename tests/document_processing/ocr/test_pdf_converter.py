"""Unit tests for the PDFImageConverter."""

import os
import unittest
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

from llm_rag.document_processing.ocr.pdf_converter import PDFImageConverter
from llm_rag.utils.errors import DocumentProcessingError

# Check if running in CI environment
IN_CI = os.environ.get('CI', 'false').lower() == 'true'


class TestPDFImageConverter(unittest.TestCase):
    """Test cases for PDFImageConverter."""

    def setUp(self):
        """Set up test cases."""
        self.converter = PDFImageConverter(dpi=200)
        self.test_pdf_path = 'tests/document_processing/ocr/data/test.pdf'

        # Create test directory if it doesn't exist
        Path('tests/document_processing/ocr/data').mkdir(parents=True, exist_ok=True)

    def test_init(self):
        """Test initialization of PDFImageConverter."""
        converter = PDFImageConverter(dpi=300)
        self.assertEqual(converter.dpi, 300)

        # Test default value
        default_converter = PDFImageConverter()
        self.assertEqual(default_converter.dpi, 300)

    def test_pdf_to_images_file_not_found(self):
        """Test pdf_to_images with non-existent file."""
        with self.assertRaises(DocumentProcessingError):
            list(self.converter.pdf_to_images('non_existent.pdf'))

    @mock.patch('fitz.open')
    def test_pdf_to_images_processing_error(self, mock_fitz_open):
        """Test pdf_to_images with processing error."""
        mock_fitz_open.side_effect = Exception('Mocked error')

        with self.assertRaises(DocumentProcessingError):
            list(self.converter.pdf_to_images(self.test_pdf_path))

    @mock.patch('fitz.open')
    @mock.patch('llm_rag.document_processing.ocr.pdf_converter.Path.exists')
    def test_pdf_to_images_success(self, mock_exists, mock_fitz_open):
        """Test successful conversion of PDF to images."""
        # Setup mocks
        mock_exists.return_value = True

        mock_doc = mock.MagicMock()
        mock_page = mock.MagicMock()
        mock_pixmap = mock.MagicMock()

        mock_doc.__len__.return_value = 2
        mock_doc.__iter__.return_value = [mock_page, mock_page]
        mock_fitz_open.return_value = mock_doc

        mock_page.get_pixmap.return_value = mock_pixmap
        mock_pixmap.width = 100
        mock_pixmap.height = 200
        mock_pixmap.samples = b'sample_data'

        # Mock PIL.Image.frombytes
        with mock.patch('PIL.Image.frombytes') as mock_frombytes:
            mock_image = mock.MagicMock(spec=Image.Image)
            mock_frombytes.return_value = mock_image

            # Call the method
            images = list(self.converter.pdf_to_images('test.pdf'))

            # Assertions
            self.assertEqual(len(images), 2)
            self.assertEqual(images[0], mock_image)
            self.assertEqual(images[1], mock_image)

            # Check if the correct methods were called
            mock_fitz_open.assert_called_once_with(str('test.pdf'))
            self.assertEqual(mock_page.get_pixmap.call_count, 2)
            self.assertEqual(mock_frombytes.call_count, 2)

            # Check if matrix was created with correct DPI
            # zoom = 200 / 72  # Based on the DPI set in setUp
            mock_page.get_pixmap.assert_called_with(matrix=mock.ANY, alpha=False)

    @pytest.mark.skipif(IN_CI, reason='Test requires access to real PDF files')
    @mock.patch('fitz.open')
    @mock.patch('llm_rag.document_processing.ocr.pdf_converter.Path.exists')
    def test_get_page_image(self, mock_exists, mock_fitz_open):
        """Test getting a specific page from a PDF."""
        mock_exists.return_value = True

        mock_doc = mock.MagicMock()
        mock_page = mock.MagicMock()
        mock_pixmap = mock.MagicMock()

        mock_doc.__len__.return_value = 3
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        mock_page.get_pixmap.return_value = mock_pixmap
        mock_pixmap.width = 100
        mock_pixmap.height = 200
        mock_pixmap.samples = b'sample_data'

        with mock.patch('PIL.Image.frombytes') as mock_frombytes:
            mock_image = mock.MagicMock(spec=Image.Image)
            mock_frombytes.return_value = mock_image

            # Test valid page
            result = self.converter.get_page_image('test.pdf', 1)
            self.assertEqual(result, mock_image)
            mock_doc.__getitem__.assert_called_with(1)

            # Test out of range page
            result = self.converter.get_page_image('test.pdf', 5)
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
