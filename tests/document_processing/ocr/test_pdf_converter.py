"""Unit tests for the PDF to Image converter module."""

import unittest
from unittest.mock import MagicMock, Mock, patch

import fitz
from PIL import Image

from llm_rag.document_processing.ocr.pdf_converter import PDFImageConverter, PDFImageConverterConfig
from llm_rag.utils.errors import DataAccessError, DocumentProcessingError


class TestPDFImageConverter(unittest.TestCase):
    """Test cases for the PDFImageConverter class."""

    def setUp(self):
        """Set up test fixtures for PDF converter tests."""
        self.test_pdf_path = "test.pdf"
        self.config = PDFImageConverterConfig(
            dpi=150,
            output_format="png",
            use_alpha_channel=False,
        )
        self.converter = PDFImageConverter(config=self.config)

    @patch("fitz.open")
    @patch("os.path.isfile")
    def test_open_pdf_document_success(self, mock_isfile, mock_open):
        """Test opening a PDF document successfully."""
        # Setup mocks
        mock_isfile.return_value = True
        mock_doc = Mock()
        mock_open.return_value = mock_doc

        # Call the method
        result = self.converter._open_pdf_document(self.test_pdf_path)

        # Assertions
        mock_isfile.assert_called_once_with(self.test_pdf_path)
        mock_open.assert_called_once_with(self.test_pdf_path)
        self.assertEqual(result, mock_doc)

    @patch("os.path.isfile")
    def test_open_pdf_document_file_not_found(self, mock_isfile):
        """Test opening a PDF document when the file doesn't exist."""
        # Setup mock
        mock_isfile.return_value = False

        # Assertions
        with self.assertRaises(DataAccessError) as context:
            self.converter._open_pdf_document(self.test_pdf_path)

        # Verify error details
        self.assertIn("PDF file not found", str(context.exception))

    @patch("fitz.open")
    @patch("os.path.isfile")
    def test_open_pdf_document_open_error(self, mock_isfile, mock_open):
        """Test opening a PDF document when an error occurs during opening."""
        # Setup mocks
        mock_isfile.return_value = True
        mock_open.side_effect = Exception("File cannot be opened")

        # Assertions
        with self.assertRaises(DataAccessError) as context:
            self.converter._open_pdf_document(self.test_pdf_path)

        # Verify error details
        self.assertIn("Failed to open PDF file", str(context.exception))

    @patch("llm_rag.document_processing.ocr.pdf_converter.PDFImageConverter._open_pdf_document")
    def test_get_pdf_page_count(self, mock_open_pdf):
        """Test getting the page count of a PDF document."""
        # Setup mock
        mock_doc = Mock()
        mock_doc.page_count = 5
        mock_open_pdf.return_value = mock_doc

        # Call the method
        result = self.converter.get_pdf_page_count(self.test_pdf_path)

        # Assertions
        self.assertEqual(result, 5)
        mock_doc.close.assert_called_once()

    @patch.object(PDFImageConverter, "_render_page_to_image")
    @patch.object(PDFImageConverter, "_open_pdf_document")
    def test_convert_pdf_page_to_image(self, mock_open_pdf, mock_render):
        """Test converting a specific PDF page to an image."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.page_count = 3
        mock_open_pdf.return_value = mock_doc

        mock_page = Mock()
        mock_doc.load_page.return_value = mock_page

        mock_image = MagicMock(spec=Image.Image)
        mock_render.return_value = mock_image

        # Call the method
        result = self.converter.convert_pdf_page_to_image(self.test_pdf_path, 2)

        # Assertions
        mock_open_pdf.assert_called_once_with(self.test_pdf_path)
        mock_doc.load_page.assert_called_once_with(1)  # 0-indexed internally
        mock_render.assert_called_once_with(mock_page)
        self.assertEqual(result, mock_image)
        mock_doc.close.assert_called_once()

    @patch.object(PDFImageConverter, "_open_pdf_document")
    def test_convert_pdf_page_to_image_invalid_page(self, mock_open_pdf):
        """Test exception when converting a page that doesn't exist."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.page_count = 3
        mock_open_pdf.return_value = mock_doc

        # Assertions for page number too high
        with self.assertRaises(DataAccessError) as context:
            self.converter.convert_pdf_page_to_image(self.test_pdf_path, 5)
        self.assertIn("Invalid page number", str(context.exception))

        # Assertions for page number too low
        with self.assertRaises(DataAccessError) as context:
            self.converter.convert_pdf_page_to_image(self.test_pdf_path, 0)
        self.assertIn("Invalid page number", str(context.exception))

        # Verify document was closed in both cases
        self.assertEqual(mock_doc.close.call_count, 2)

    @patch.object(PDFImageConverter, "_render_page_to_image")
    @patch.object(PDFImageConverter, "_open_pdf_document")
    def test_convert_pdf_to_images(self, mock_open_pdf, mock_render):
        """Test converting multiple PDF pages to images."""
        # Setup mocks
        mock_doc = Mock()
        mock_doc.page_count = 3
        mock_open_pdf.return_value = mock_doc

        mock_pages = [Mock() for _ in range(3)]
        mock_doc.load_page.side_effect = mock_pages

        mock_images = [MagicMock(spec=Image.Image) for _ in range(3)]
        mock_render.side_effect = mock_images

        # Call the method and collect results
        results = list(self.converter.convert_pdf_to_images(self.test_pdf_path))

        # Assertions
        mock_open_pdf.assert_called_once_with(self.test_pdf_path)
        self.assertEqual(mock_doc.load_page.call_count, 3)
        self.assertEqual(mock_render.call_count, 3)
        self.assertEqual(len(results), 3)

        # Check returned page numbers (0-indexed) and images
        for i, (page_num, image) in enumerate(results):
            self.assertEqual(page_num, i)
            self.assertEqual(image, mock_images[i])

        mock_doc.close.assert_called_once()

    @patch.object(PDFImageConverter, "_open_pdf_document")
    def test_convert_pdf_to_images_with_page_range(self, mock_open_pdf):
        """Test converting a specific range of PDF pages to images."""
        # Setup
        config = PDFImageConverterConfig(
            first_page=2,  # 1-indexed in the API
            last_page=4,  # 1-indexed in the API
        )
        converter = PDFImageConverter(config=config)

        # Setup mocks
        mock_doc = Mock()
        mock_doc.page_count = 5
        mock_open_pdf.return_value = mock_doc

        mock_pages = [Mock() for _ in range(3)]  # Pages 1-3 (0-indexed)
        mock_doc.load_page.side_effect = mock_pages

        # Mock the render method to return test images
        converter._render_page_to_image = MagicMock()
        test_images = [MagicMock(spec=Image.Image) for _ in range(3)]
        converter._render_page_to_image.side_effect = test_images

        # Call the method and collect results
        results = list(converter.convert_pdf_to_images(self.test_pdf_path))

        # Assertions
        self.assertEqual(len(results), 3)  # Should get 3 pages (2-4)
        expected_pages = [1, 2, 3]  # 0-indexed internally

        # Check that the correct pages were loaded and the correct images returned
        for i, (page_num, image) in enumerate(results):
            self.assertEqual(page_num, expected_pages[i])
            self.assertEqual(image, test_images[i])
            mock_doc.load_page.assert_any_call(expected_pages[i])

    def test_render_page_to_image(self):
        """Test rendering a PDF page to an image."""
        # Since we can't easily mock fitz.Page, this is more of an integration test
        # For a proper unit test, we'd need to create a test fixture or mock at a lower level
        with patch.object(fitz.Page, "get_pixmap") as mock_get_pixmap:
            # Create a mock pixmap
            mock_pixmap = Mock()
            mock_pixmap.width = 100
            mock_pixmap.height = 100
            mock_pixmap.samples = b"\x00" * (100 * 100 * 3)  # RGB data
            mock_get_pixmap.return_value = mock_pixmap

            # Create a mock page
            mock_page = Mock(spec=fitz.Page)
            mock_page.get_pixmap = mock_get_pixmap
            mock_page.number = 1

            # Create test data for the Matrix constructor
            with patch("fitz.Matrix") as mock_matrix_class:
                mock_matrix = Mock()
                mock_matrix_class.return_value = mock_matrix

                # With Image.frombytes mocked to return a test image
                test_image = MagicMock(spec=Image.Image)
                with patch.object(Image, "frombytes", return_value=test_image) as mock_frombytes:
                    # Call the method
                    result = self.converter._render_page_to_image(mock_page)

                    # Assertions
                    zoom = self.config.dpi / 72  # Expected zoom factor
                    mock_matrix_class.assert_called_once_with(zoom, zoom)
                    mock_get_pixmap.assert_called_once_with(matrix=mock_matrix, alpha=self.config.use_alpha_channel)
                    mock_frombytes.assert_called_once_with(
                        "RGB", [mock_pixmap.width, mock_pixmap.height], mock_pixmap.samples
                    )
                    self.assertEqual(result, test_image)

    def test_render_page_to_image_with_alpha(self):
        """Test rendering a PDF page to an image with alpha channel."""
        # Setup
        config = PDFImageConverterConfig(use_alpha_channel=True)
        converter = PDFImageConverter(config=config)

        # Since we can't easily mock fitz.Page, similar approach to previous test
        with patch.object(fitz.Page, "get_pixmap") as mock_get_pixmap:
            # Create a mock pixmap with alpha
            mock_pixmap = Mock()
            mock_pixmap.width = 100
            mock_pixmap.height = 100
            mock_pixmap.samples = b"\x00" * (100 * 100 * 4)  # RGBA data
            mock_get_pixmap.return_value = mock_pixmap

            # Create a mock page
            mock_page = Mock(spec=fitz.Page)
            mock_page.get_pixmap = mock_get_pixmap
            mock_page.number = 1

            # Create test data for the Matrix constructor
            with patch("fitz.Matrix") as mock_matrix_class:
                mock_matrix = Mock()
                mock_matrix_class.return_value = mock_matrix

                # With Image.frombytes mocked to return a test image
                test_image = MagicMock(spec=Image.Image)
                with patch.object(Image, "frombytes", return_value=test_image) as mock_frombytes:
                    # Call the method
                    result = converter._render_page_to_image(mock_page)

                    # Assertions
                    mock_get_pixmap.assert_called_once_with(matrix=mock_matrix, alpha=True)
                    mock_frombytes.assert_called_once_with(
                        "RGBA", [mock_pixmap.width, mock_pixmap.height], mock_pixmap.samples
                    )
                    self.assertEqual(result, test_image)

    @patch.object(fitz.Page, "get_pixmap")
    def test_render_page_to_image_error(self, mock_get_pixmap):
        """Test handling of errors during page rendering."""
        # Setup
        mock_get_pixmap.side_effect = Exception("Rendering error")
        mock_page = Mock(spec=fitz.Page)
        mock_page.number = 2

        # Create test data for the Matrix constructor
        with patch("fitz.Matrix"):
            # Assertions
            with self.assertRaises(DocumentProcessingError) as context:
                self.converter._render_page_to_image(mock_page)

            # Verify error message contains page number
            error_message = str(context.exception)
            self.assertIn("Error rendering page", error_message)
            self.assertIn("2", error_message)


if __name__ == "__main__":
    unittest.main()
