"""Unit tests for the OCR engine module."""

import subprocess
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from llm_rag.document_processing.ocr.ocr_engine import (
    OCROutputFormat,
    PageSegmentationMode,
    TesseractConfig,
    TesseractOCREngine,
)
from llm_rag.utils.errors import DocumentProcessingError, ExternalServiceError


class TestTesseractOCREngine(unittest.TestCase):
    """Test cases for the TesseractOCREngine class."""

    def setUp(self):
        """Set up test fixtures for OCR engine tests."""
        # Create a test image
        self.test_image = MagicMock(spec=Image.Image)
        self.test_image.width = 1000
        self.test_image.height = 800

        # Default config with German language
        self.config = TesseractConfig(
            languages=["deu"],
            psm=PageSegmentationMode.AUTO,
            timeout=10,
        )

        # Patch the check_tesseract_installed method to avoid actual system calls
        with patch.object(TesseractOCREngine, "_check_tesseract_installed"):
            self.ocr_engine = TesseractOCREngine(config=self.config)

    @patch("pytesseract.get_tesseract_version")
    def test_check_tesseract_installed_success(self, mock_get_version):
        """Test successful check of tesseract installation."""
        # Setup mock
        mock_get_version.return_value = "4.1.1"

        # Remove patching from setUp to test the actual method
        engine = TesseractOCREngine(config=self.config)
        engine._check_tesseract_installed = TesseractOCREngine._check_tesseract_installed.__get__(engine)

        # Call should not raise an exception
        engine._check_tesseract_installed()

        # Check that version is logged correctly
        mock_get_version.assert_called()

    @patch("pytesseract.get_tesseract_version")
    def test_check_tesseract_installed_failure(self, mock_get_version):
        """Test handling of missing tesseract installation."""
        # Setup mock to simulate missing tesseract
        mock_get_version.side_effect = Exception("Tesseract not found")

        # Create new engine with real check method
        engine = TesseractOCREngine.__new__(TesseractOCREngine)
        engine.config = self.config

        # The check should raise an exception
        with self.assertRaises(ExternalServiceError) as context:
            TesseractOCREngine._check_tesseract_installed(engine)

        # Check error details
        self.assertIn("Tesseract OCR is not available", str(context.exception))

    def test_build_config_string(self):
        """Test building the tesseract configuration string."""
        # Default config
        config_str = self.ocr_engine._build_config_string()
        self.assertEqual(config_str, f"--psm {PageSegmentationMode.AUTO.value} --oem 3")

        # With custom config
        engine = TesseractOCREngine(
            config=TesseractConfig(
                psm=PageSegmentationMode.SINGLE_COLUMN,
                oem=1,
                custom_config="--dpi 300 --tessdata-dir /usr/share/tessdata",
            )
        )
        config_str = engine._build_config_string()
        self.assertEqual(
            config_str,
            f"--psm {PageSegmentationMode.SINGLE_COLUMN.value} --oem 1 --dpi 300 --tessdata-dir /usr/share/tessdata",
        )

    @patch("pytesseract.image_to_string")
    def test_image_to_text(self, mock_image_to_string):
        """Test basic OCR text extraction."""
        # Setup mock
        expected_text = "Sample German text: Der schnelle braune Fuchs springt über den faulen Hund."
        mock_image_to_string.return_value = expected_text

        # Call method
        result = self.ocr_engine.image_to_text(self.test_image)

        # Assertions
        self.assertEqual(result, expected_text)
        mock_image_to_string.assert_called_once_with(
            self.test_image,
            lang="deu",
            config="--psm 3 --oem 3",
            timeout=10,
        )

    @patch("pytesseract.image_to_string")
    def test_image_to_text_failure(self, mock_image_to_string):
        """Test handling of OCR failures."""
        # Setup mock to simulate OCR failure
        mock_image_to_string.side_effect = Exception("OCR processing failed")

        # Assertions
        with self.assertRaises(DocumentProcessingError) as context:
            self.ocr_engine.image_to_text(self.test_image)

        # Check error details
        self.assertIn("OCR text extraction failed", str(context.exception))

    @patch("pytesseract.image_to_string")
    def test_image_to_data_text_format(self, mock_image_to_string):
        """Test OCR data extraction with text format."""
        # Setup mock
        expected_text = "Sample text"
        mock_image_to_string.return_value = expected_text

        # Call method
        result = self.ocr_engine.image_to_data(self.test_image, output_format=OCROutputFormat.TEXT)

        # Assertions
        self.assertEqual(result, expected_text)
        mock_image_to_string.assert_called_once_with(
            self.test_image,
            lang="deu",
            config="--psm 3 --oem 3",
            timeout=10,
        )

    @patch("pytesseract.image_to_pdf_or_hocr")
    def test_image_to_data_hocr_format(self, mock_image_to_hocr):
        """Test OCR data extraction with hOCR format."""
        # Setup mock
        expected_hocr = "<html><body>hOCR content</body></html>"
        mock_image_to_hocr.return_value = expected_hocr

        # Call method
        result = self.ocr_engine.image_to_data(self.test_image, output_format=OCROutputFormat.HOCR)

        # Assertions
        self.assertEqual(result, expected_hocr)
        mock_image_to_hocr.assert_called_once_with(
            self.test_image,
            lang="deu",
            config="--psm 3 --oem 3",
            extension="hocr",
            timeout=10,
        )

    @patch("pytesseract.image_to_data")
    def test_image_to_data_tsv_format(self, mock_image_to_data):
        """Test OCR data extraction with TSV format."""
        # Setup mock
        expected_tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
        mock_image_to_data.return_value = expected_tsv

        # Call method
        result = self.ocr_engine.image_to_data(self.test_image, output_format=OCROutputFormat.TSV)

        # Assertions
        self.assertEqual(result, expected_tsv)
        mock_image_to_data.assert_called_once_with(
            self.test_image,
            lang="deu",
            config="--psm 3 --oem 3",
            timeout=10,
        )

    @patch("pytesseract.image_to_data")
    def test_image_to_data_json_format(self, mock_image_to_data):
        """Test OCR data extraction with JSON format."""
        # Setup mock
        expected_dict = {
            "level": [1, 2, 3],
            "page_num": [1, 1, 1],
            "text": ["line1", "line2", "line3"],
            "conf": [90.5, 85.2, 95.0],
        }
        mock_image_to_data.return_value = expected_dict

        # Call method
        result = self.ocr_engine.image_to_data(self.test_image, output_format=OCROutputFormat.JSON)

        # Assertions
        self.assertEqual(result, expected_dict)
        mock_image_to_data.assert_called_once_with(
            self.test_image,
            lang="deu",
            config="--psm 3 --oem 3",
            output_type=pytest.importorskip("pytesseract").Output.DICT,
            timeout=10,
        )

    @patch("pytesseract.image_to_alto_xml")
    def test_image_to_data_alto_format(self, mock_image_to_alto):
        """Test OCR data extraction with ALTO XML format."""
        # Setup mock
        expected_alto = "<alto><Layout>ALTO XML content</Layout></alto>"
        mock_image_to_alto.return_value = expected_alto

        # Call method
        result = self.ocr_engine.image_to_data(self.test_image, output_format=OCROutputFormat.ALTO)

        # Assertions
        self.assertEqual(result, expected_alto)
        mock_image_to_alto.assert_called_once_with(
            self.test_image,
            lang="deu",
            config="--psm 3 --oem 3",
            timeout=10,
        )

    def test_image_to_data_invalid_format(self):
        """Test handling of invalid output format."""
        # Create a mock format that's not in the enum
        invalid_format = Mock()
        invalid_format.name = "INVALID"

        # Assertions
        with self.assertRaises(ValueError) as context:
            self.ocr_engine.image_to_data(self.test_image, output_format=invalid_format)

        # Check error message
        self.assertIn("Unsupported output format", str(context.exception))

    @patch("subprocess.check_output")
    def test_get_supported_languages(self, mock_check_output):
        """Test getting supported languages from Tesseract."""
        # Setup mock
        mock_output = "List of available languages (3):\ndeu\neng\nfra\n"
        mock_check_output.return_value = mock_output

        # Call method
        result = self.ocr_engine.get_supported_languages()

        # Assertions
        self.assertEqual(result, ["deu", "eng", "fra"])
        mock_check_output.assert_called_once_with(
            ["tesseract", "--list-langs"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=5,
        )

    @patch("subprocess.check_output")
    def test_get_supported_languages_with_custom_path(self, mock_check_output):
        """Test getting supported languages with custom tesseract path."""
        # Setup
        custom_path = "/usr/local/bin/tesseract"
        config = TesseractConfig(tesseract_cmd=custom_path)

        # Create new engine with custom path
        with patch.object(TesseractOCREngine, "_check_tesseract_installed"):
            engine = TesseractOCREngine(config=config)

        # Setup mock
        mock_output = "List of available languages (2):\ndeu\neng\n"
        mock_check_output.return_value = mock_output

        # Call method
        result = engine.get_supported_languages()

        # Assertions
        self.assertEqual(result, ["deu", "eng"])
        mock_check_output.assert_called_once_with(
            [custom_path, "--list-langs"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=5,
        )

    @patch("subprocess.check_output")
    def test_get_supported_languages_failure(self, mock_check_output):
        """Test handling of failures when getting supported languages."""
        # Setup mock to simulate command failure
        mock_check_output.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["tesseract", "--list-langs"], output=b"Error: Cannot list languages"
        )

        # Assertions
        with self.assertRaises(ExternalServiceError) as context:
            self.ocr_engine.get_supported_languages()

        # Check error details
        self.assertIn("Failed to get supported languages", str(context.exception))


if __name__ == "__main__":
    unittest.main()
