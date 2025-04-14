"""Integration tests for the PDF image converter module."""

import unittest
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from llm_rag.document_processing.ocr.pdf_converter import PDFImageConverter
from llm_rag.utils.errors import DataAccessError


class TestPDFImageConverterIntegration(unittest.TestCase):
    """Integration tests for the PDF image converter."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Create a test PDF file
        cls.test_dir = Path("tests/data/pdfs")
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        cls.valid_pdf_path = cls.test_dir / "valid_test.pdf"
        cls.create_test_pdf(cls.valid_pdf_path)

        cls.invalid_pdf_path = cls.test_dir / "invalid_test.pdf"
        with open(cls.invalid_pdf_path, "w") as f:
            f.write("This is not a valid PDF file")

        cls.nonexistent_pdf_path = cls.test_dir / "nonexistent.pdf"

        # Create a converter instance
        cls.converter = PDFImageConverter()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests have run."""
        # Remove test PDF files
        if cls.valid_pdf_path.exists():
            cls.valid_pdf_path.unlink()
        if cls.invalid_pdf_path.exists():
            cls.invalid_pdf_path.unlink()

    @classmethod
    def create_test_pdf(cls, path):
        """Create a simple test PDF file."""
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)  # Letter size

        # Add some text to the page
        text_rect = fitz.Rect(100, 100, 500, 150)
        page.insert_text(text_rect.tl, "This is a test PDF document for OCR testing", fontsize=12)

        # Add some shapes
        page.draw_rect(fitz.Rect(100, 200, 500, 250), color=(0, 0, 1), fill=(1, 1, 0))
        page.draw_circle(fitz.Point(300, 350), 50, color=(1, 0, 0))

        # Save the PDF
        doc.save(str(path))
        doc.close()

    def test_convert_valid_pdf_to_images(self):
        """Test converting a valid PDF to images."""
        # Get images from each page
        images = list(self.converter.convert_pdf_to_images(self.valid_pdf_path))

        # Check that we got images
        self.assertEqual(len(images), 1)  # One page in our test PDF

        # Check each image
        for page_num, image in images:
            self.assertEqual(page_num, 0)  # First and only page
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.mode, "RGB")
            self.assertTrue(image.width > 0)
            self.assertTrue(image.height > 0)

    def test_get_pdf_page_count(self):
        """Test getting page count from a valid PDF."""
        count = self.converter.get_pdf_page_count(self.valid_pdf_path)
        self.assertEqual(count, 1)  # One page in our test PDF

    def test_convert_specific_page(self):
        """Test converting a specific page from PDF."""
        image = self.converter.convert_pdf_page_to_image(self.valid_pdf_path, 1)  # 1-indexed
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")

    def test_nonexistent_pdf(self):
        """Test handling a nonexistent PDF file."""
        with self.assertRaises(DataAccessError):
            self.converter.get_pdf_page_count(self.nonexistent_pdf_path)

    def test_invalid_pdf(self):
        """Test handling an invalid PDF file."""
        with self.assertRaises(DataAccessError):
            self.converter.get_pdf_page_count(self.invalid_pdf_path)

    def test_invalid_page_number(self):
        """Test handling an invalid page number."""
        with self.assertRaises(DataAccessError):
            self.converter.convert_pdf_page_to_image(self.valid_pdf_path, 999)  # Non-existent page

    def test_convert_with_custom_dpi(self):
        """Test converting with custom DPI setting."""
        # Standard DPI
        image_standard = self.converter.convert_pdf_page_to_image(self.valid_pdf_path, 1)

        # Higher DPI
        high_dpi_converter = PDFImageConverter()
        image_high_dpi = high_dpi_converter._render_page_to_image(self.valid_pdf_path, page_num=0, dpi=600)

        # Higher DPI should result in a larger image
        self.assertGreater(image_high_dpi.width, image_standard.width)
        self.assertGreater(image_high_dpi.height, image_standard.height)


if __name__ == "__main__":
    unittest.main()
