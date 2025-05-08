"""Integration tests for the OCR pipeline."""

import unittest
from pathlib import Path

from llm_rag.document_processing.ocr.pipeline import OCRPipeline


class TestOCRIntegration(unittest.TestCase):
    """Integration tests for the OCR pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Use an existing test PDF that's small in size
        cls.test_pdf_path = Path('data/documents/test_subset/VDE_0636-3_A2_E__DIN_VDE_0636-3_A2__2010-04.pdf')

        # Skip test if the file doesn't exist
        if not cls.test_pdf_path.exists():
            raise unittest.SkipTest(f'Test PDF not found at {cls.test_pdf_path}')

    def test_basic_ocr_pipeline(self):
        """Test the complete OCR pipeline with a real PDF."""
        # Skip test if PyTesseract is not properly configured
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
        except (ImportError, pytesseract.TesseractNotFoundError):
            self.skipTest('Tesseract not properly installed or configured')

        # Initialize the OCR pipeline
        pipeline = OCRPipeline()

        # Process the test PDF - just process the first page to keep the test quick
        pipeline.config.process_pages = [0]

        try:
            result = pipeline.process_pdf(self.test_pdf_path)

            # Basic assertions to verify something was extracted
            self.assertIsNotNone(result)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

            # Print first 200 chars of result for debugging
            print(f'OCR Result (first 200 chars): {result[:200]}')

        except Exception as e:
            self.fail(f'OCR pipeline failed with error: {str(e)}')

    def test_page_specific_processing(self):
        """Test processing specific pages from a PDF."""
        # Skip test if PyTesseract is not properly configured
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
        except (ImportError, pytesseract.TesseractNotFoundError):
            self.skipTest('Tesseract not properly installed or configured')

        # Initialize the OCR pipeline
        pipeline = OCRPipeline()

        # Process only page 1 (second page)
        try:
            result = pipeline.process_pdf_pages(self.test_pdf_path, [1])

            # Basic assertions
            self.assertIsNotNone(result)
            self.assertIn(1, result)
            self.assertIsInstance(result[1], str)
            self.assertGreater(len(result[1]), 0)

            # Print first 200 chars of result for debugging
            if result and 1 in result:
                print(f'Page 1 OCR Result (first 200 chars): {result[1][:200]}')

        except Exception as e:
            self.fail(f'OCR pipeline failed with error: {str(e)}')


if __name__ == '__main__':
    unittest.main()
