"""Unit tests for the OCR pipeline."""

import unittest
from unittest import mock

from PIL import Image

from llm_rag.document_processing.ocr.pipeline import OCRPipeline, OCRPipelineConfig
from llm_rag.utils.errors import DocumentProcessingError


class TestOCRPipeline(unittest.TestCase):
    """Test cases for OCRPipeline."""

    def setUp(self):
        """Set up test cases."""
        # Create OCR pipeline with default configuration
        self.pipeline = OCRPipeline()

        # Create a mock image for testing
        self.mock_image = mock.MagicMock(spec=Image.Image)

        # Mock PDF path for testing
        self.test_pdf_path = 'test.pdf'

    def test_init(self):
        """Test initialization of OCRPipeline."""
        # Test with default configuration
        pipeline = OCRPipeline()
        self.assertIsNotNone(pipeline.config)
        self.assertIsNotNone(pipeline.pdf_converter)
        self.assertIsNotNone(pipeline.ocr_engine)

        # Test with custom configuration
        custom_config = OCRPipelineConfig(pdf_dpi=400, languages=['eng', 'deu'], psm=1, oem=2)
        pipeline = OCRPipeline(config=custom_config)
        self.assertEqual(pipeline.config, custom_config)
        self.assertEqual(pipeline.pdf_converter.dpi, 400)
        self.assertEqual(pipeline.ocr_engine.languages, 'eng+deu')
        self.assertEqual(pipeline.ocr_engine.psm, 1)
        self.assertEqual(pipeline.ocr_engine.oem, 2)

    @mock.patch('llm_rag.document_processing.ocr.pdf_converter.PDFImageConverter.pdf_to_images')
    @mock.patch('llm_rag.document_processing.ocr.ocr_engine.TesseractOCREngine.process_multiple_images')
    def test_process_pdf(self, mock_process_images, mock_pdf_to_images):
        """Test processing a PDF document."""
        # Setup mocks
        mock_pdf_to_images.return_value = [self.mock_image, self.mock_image]
        mock_process_images.return_value = ['Page 1 text', 'Page 2 text']

        # Set output format to raw text for this test
        self.pipeline.config.output_format = 'raw'

        # Process PDF
        result = self.pipeline.process_pdf(self.test_pdf_path)

        # Assertions
        self.assertEqual(result, 'Page 1 text\n\nPage 2 text')
        mock_pdf_to_images.assert_called_once()
        mock_process_images.assert_called_once_with([self.mock_image, self.mock_image])

    @mock.patch('llm_rag.document_processing.ocr.pdf_converter.PDFImageConverter.pdf_to_images')
    @mock.patch('llm_rag.document_processing.ocr.ocr_engine.TesseractOCREngine.process_multiple_images')
    def test_process_pdf_no_images(self, mock_process_images, mock_pdf_to_images):
        """Test processing a PDF document that yields no valid images."""
        # Setup mocks
        mock_pdf_to_images.return_value = []

        # Assertions
        with self.assertRaises(DocumentProcessingError):
            self.pipeline.process_pdf(self.test_pdf_path)

        mock_pdf_to_images.assert_called_once()
        mock_process_images.assert_not_called()

    @mock.patch('llm_rag.document_processing.ocr.pdf_converter.PDFImageConverter.get_page_image')
    @mock.patch('llm_rag.document_processing.ocr.ocr_engine.TesseractOCREngine.process_image')
    def test_process_pdf_pages(self, mock_process_image, mock_get_page_image):
        """Test processing specific pages of a PDF document."""
        # Setup mocks
        mock_get_page_image.return_value = self.mock_image
        mock_process_image.return_value = 'Page text'

        # Process pages
        result = self.pipeline.process_pdf_pages(self.test_pdf_path, [0, 2])

        # Assertions
        self.assertEqual(result, {0: 'Page text', 2: 'Page text'})
        self.assertEqual(mock_get_page_image.call_count, 2)
        self.assertEqual(mock_process_image.call_count, 2)

    @mock.patch('llm_rag.document_processing.ocr.pdf_converter.PDFImageConverter.get_page_image')
    @mock.patch('llm_rag.document_processing.ocr.ocr_engine.TesseractOCREngine.process_image')
    def test_process_pdf_pages_none_valid(self, mock_process_image, mock_get_page_image):
        """Test processing specific pages of a PDF document when no pages are valid."""
        # Setup mocks
        mock_get_page_image.return_value = None

        # Assertions
        with self.assertRaises(DocumentProcessingError):
            self.pipeline.process_pdf_pages(self.test_pdf_path, [0, 2])

        self.assertEqual(mock_get_page_image.call_count, 2)
        mock_process_image.assert_not_called()

    @mock.patch('llm_rag.document_processing.ocr.pdf_converter.PDFImageConverter.get_page_image')
    def test_process_pdf_pages_with_some_invalid(self, mock_get_page_image):
        """Test processing specific pages when some pages are invalid."""
        # Setup mocks to return image for page 0 but None for page 2
        mock_get_page_image.side_effect = lambda path, page_num: (self.mock_image if page_num == 0 else None)

        # Create a mock for process_image that returns text
        with mock.patch.object(self.pipeline.ocr_engine, 'process_image', return_value='Page text'):
            # Process pages
            result = self.pipeline.process_pdf_pages(self.test_pdf_path, [0, 2])

            # Assertions
            self.assertEqual(result, {0: 'Page text'})
            self.assertEqual(mock_get_page_image.call_count, 2)

    @mock.patch('llm_rag.document_processing.ocr.pdf_converter.PDFImageConverter.get_page_image')
    @mock.patch('llm_rag.document_processing.ocr.ocr_engine.TesseractOCREngine.process_image')
    def test_process_pdf_pages_with_error(self, mock_process_image, mock_get_page_image):
        """Test error handling when OCR processing fails for a page."""
        # Setup mocks
        mock_get_page_image.return_value = self.mock_image
        mock_process_image.side_effect = DocumentProcessingError('OCR error')

        # Assertions
        with self.assertRaises(DocumentProcessingError):
            self.pipeline.process_pdf_pages(self.test_pdf_path, [0])


if __name__ == '__main__':
    unittest.main()
