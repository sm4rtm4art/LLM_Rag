"""OCR engine module for text extraction from images.

This module provides a wrapper for Tesseract OCR to extract text from images.
It supports various configuration options for optimizing OCR results.
"""

import os
import re
import shutil
import subprocess  # nosec B404
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import pytesseract
from PIL import Image

from llm_rag.utils.errors import DocumentProcessingError, ErrorCode, ExternalServiceError, handle_exceptions
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


class PageSegmentationMode(Enum):
    """Tesseract Page Segmentation Modes.

    These determine how Tesseract segments the page for analysis.
    For more information: https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
    """

    OSD_ONLY = 0  # Orientation and script detection only
    AUTO_OSD = 1  # Automatic page segmentation with orientation and script detection
    AUTO_ONLY = 2  # Automatic page segmentation, but no OSD or OCR
    AUTO = 3  # Fully automatic page segmentation, but no OSD (default)
    SINGLE_COLUMN = 4  # Assume a single column of text of variable sizes
    SINGLE_BLOCK_VERT_TEXT = 5  # Assume a single uniform block of vertically aligned text
    SINGLE_BLOCK = 6  # Assume a single uniform block of text
    SINGLE_LINE = 7  # Treat the image as a single text line
    SINGLE_WORD = 8  # Treat the image as a single word
    CIRCLE_WORD = 9  # Treat the image as a single word in a circle
    SINGLE_CHAR = 10  # Treat the image as a single character
    SPARSE_TEXT = 11  # Find as much text as possible in no particular order
    SPARSE_TEXT_OSD = 12  # Sparse text with orientation and script detection
    RAW_LINE = 13  # Treat the image as a single text line, bypassing hacks


class OCROutputFormat(Enum):
    """OCR output formats supported by the engine."""

    TEXT = "text"  # Plain text output
    HOCR = "hocr"  # hOCR format (HTML with position information)
    TSV = "tsv"  # Tab-separated values with confidence and position
    JSON = "json"  # JSON output with detailed information
    ALTO = "alto"  # ALTO XML format


@dataclass
class TesseractConfig:
    """Configuration for Tesseract OCR.

    Attributes:
        tesseract_cmd: Path to the Tesseract executable (default: use system PATH).
        languages: List of language codes to use for OCR (default: ["deu"]).
        psm: Page segmentation mode (default: PageSegmentationMode.AUTO).
        oem: OCR Engine Mode (0-3) (default: 3 - Default, based on what is available).
        custom_config: Additional Tesseract configuration parameters (default: None).
        timeout: Timeout in seconds for Tesseract processing (default: 30).
        output_format: Format of the OCR output (default: OCROutputFormat.TEXT).

    """

    tesseract_cmd: Optional[str] = None
    languages: List[str] = None
    psm: PageSegmentationMode = PageSegmentationMode.AUTO
    oem: int = 3
    custom_config: Optional[str] = None
    timeout: int = 30
    output_format: OCROutputFormat = OCROutputFormat.TEXT

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.languages is None:
            self.languages = ["deu"]  # German language model as default

        # Validate tesseract_cmd if provided
        if self.tesseract_cmd:
            # Ensure the path contains only valid characters
            if not re.match(r"^[a-zA-Z0-9_\-./\\]+$", self.tesseract_cmd):
                raise ValueError(
                    "tesseract_cmd contains invalid characters. Only alphanumeric, "
                    "underscore, hyphen, dot, slash, and backslash are allowed."
                )
            # Set the Tesseract command if it exists and is accessible
            if not os.path.isfile(self.tesseract_cmd) and not shutil.which(self.tesseract_cmd):
                raise ValueError(f"tesseract_cmd '{self.tesseract_cmd}' is not a valid executable path")
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd


class TesseractOCREngine:
    """OCR engine that uses Tesseract for text extraction from images.

    This class provides a wrapper around the pytesseract library with
    configurable options for different OCR scenarios.
    """

    def __init__(self, config: Optional[TesseractConfig] = None):
        """Initialize the Tesseract OCR engine.

        Args:
            config: Configuration for the Tesseract OCR engine. If None,
                default configuration will be used.

        """
        self.config = config or TesseractConfig()
        self._check_tesseract_installed()

        logger.info(
            f"Initialized TesseractOCREngine with languages={self.config.languages}, psm={self.config.psm.value}"
        )

    def _check_tesseract_installed(self) -> None:
        """Check if Tesseract is installed and properly configured.

        Raises:
            ExternalServiceError: If Tesseract is not available.

        """
        try:
            pytesseract.get_tesseract_version()
            logger.debug(f"Tesseract version: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            error_msg = (
                "Tesseract OCR is not available. Ensure it is installed and "
                "properly configured, or specify the path using the tesseract_cmd "
                "configuration option."
            )
            logger.error(error_msg)
            raise ExternalServiceError(
                error_msg,
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                original_exception=e,
            ) from e

    def _build_config_string(self) -> str:
        """Build the Tesseract configuration string.

        Returns:
            A configuration string for Tesseract.

        """
        config_parts = [
            f"--psm {self.config.psm.value}",
            f"--oem {self.config.oem}",
        ]

        if self.config.custom_config:
            config_parts.append(self.config.custom_config)

        return " ".join(config_parts)

    @handle_exceptions(
        error_type=DocumentProcessingError,
        error_code=ErrorCode.DOCUMENT_PARSE_ERROR,
        default_message="Failed to perform OCR on image",
    )
    def image_to_text(self, image: Image.Image) -> str:
        """Extract plain text from an image using OCR.

        Args:
            image: PIL Image object to process.

        Returns:
            Extracted text from the image.

        Raises:
            DocumentProcessingError: If OCR processing fails.

        """
        logger.debug("Performing OCR on image")

        try:
            lang = "+".join(self.config.languages)
            config = self._build_config_string()

            text = pytesseract.image_to_string(
                image,
                lang=lang,
                config=config,
                timeout=self.config.timeout,
            )

            logger.debug(f"OCR completed, extracted {len(text)} characters")
            return text
        except Exception as e:
            raise DocumentProcessingError(
                "OCR text extraction failed",
                error_code=ErrorCode.DOCUMENT_PARSE_ERROR,
                original_exception=e,
                details={"image_size": f"{image.width}x{image.height}"},
            ) from e

    @handle_exceptions(
        error_type=DocumentProcessingError,
        error_code=ErrorCode.DOCUMENT_PARSE_ERROR,
        default_message="Failed to perform OCR data extraction",
    )
    def image_to_data(self, image: Image.Image, output_format: Optional[OCROutputFormat] = None) -> Union[str, Dict]:
        """Extract detailed OCR data from an image.

        Args:
            image: PIL Image object to process.
            output_format: Output format (overrides config if provided).

        Returns:
            Extracted OCR data in the specified format.

        Raises:
            DocumentProcessingError: If OCR processing fails.

        """
        logger.debug("Extracting OCR data from image")

        output_format = output_format or self.config.output_format
        lang = "+".join(self.config.languages)
        config = self._build_config_string()

        try:
            if output_format == OCROutputFormat.TEXT:
                return pytesseract.image_to_string(image, lang=lang, config=config, timeout=self.config.timeout)
            elif output_format == OCROutputFormat.HOCR:
                return pytesseract.image_to_pdf_or_hocr(
                    image, lang=lang, config=config, extension="hocr", timeout=self.config.timeout
                )
            elif output_format == OCROutputFormat.TSV:
                return pytesseract.image_to_data(image, lang=lang, config=config, timeout=self.config.timeout)
            elif output_format == OCROutputFormat.JSON:
                data = pytesseract.image_to_data(
                    image, lang=lang, config=config, output_type=pytesseract.Output.DICT, timeout=self.config.timeout
                )
                return data
            elif output_format == OCROutputFormat.ALTO:
                return pytesseract.image_to_alto_xml(image, lang=lang, config=config, timeout=self.config.timeout)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        except Exception as e:
            raise DocumentProcessingError(
                f"OCR data extraction failed for format {output_format.value}",
                error_code=ErrorCode.DOCUMENT_PARSE_ERROR,
                original_exception=e,
                details={"image_size": f"{image.width}x{image.height}"},
            ) from e

    def get_supported_languages(self) -> List[str]:
        """Get list of languages supported by the Tesseract installation.

        Returns:
            List of language codes supported by Tesseract.

        Raises:
            ExternalServiceError: If Tesseract is not available or the command fails.

        """
        try:
            # Get the tesseract command
            tesseract_cmd = self.config.tesseract_cmd or pytesseract.pytesseract.tesseract_cmd or "tesseract"

            # Validate tesseract_cmd
            if not re.match(r"^[a-zA-Z0-9_\-./\\]+$", tesseract_cmd):
                raise ValueError(
                    "tesseract_cmd contains invalid characters. Only alphanumeric, "
                    "underscore, hyphen, dot, slash, and backslash are allowed."
                )

            # Check if command exists
            tesseract_path = shutil.which(tesseract_cmd)
            if not os.path.isfile(tesseract_cmd) and not tesseract_path:
                raise ValueError(f"tesseract_cmd '{tesseract_cmd}' not found")

            # Use the resolved path if found by shutil.which
            if tesseract_path:
                tesseract_cmd = tesseract_path

            # Execute command safely with the validated path
            output = subprocess.check_output(
                [tesseract_cmd, "--list-langs"],  # nosec B603
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                timeout=5,
            )

            # Parse the output to extract language codes
            lines = output.strip().split("\n")
            # Skip the header line which usually says "List of available languages (xx):"
            languages = [line.strip() for line in lines[1:] if line.strip()]

            return languages
        except ValueError as ve:
            error_msg = f"Invalid tesseract command: {str(ve)}"
            logger.error(error_msg, exc_info=True)
            raise ExternalServiceError(
                error_msg,
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
            ) from ve
        except subprocess.SubprocessError as se:
            error_msg = f"Failed to execute tesseract: {str(se)}"
            logger.error(error_msg, exc_info=True)
            raise ExternalServiceError(
                error_msg,
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                original_exception=se,
            ) from se
        except Exception as e:
            error_msg = "Failed to get supported languages from Tesseract"
            logger.error(error_msg, exc_info=True)
            raise ExternalServiceError(
                error_msg,
                error_code=ErrorCode.SERVICE_UNAVAILABLE,
                original_exception=e,
            ) from e
