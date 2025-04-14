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

from llm_rag.utils.errors import DocumentProcessingError, ErrorCode, ExternalServiceError
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
    """OCR engine wrapper for Tesseract.

    This class provides a wrapper around the Tesseract OCR engine, with
    configuration options for language models, page segmentation mode, and
    other Tesseract settings.
    """

    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        languages: Union[str, List[str]] = "eng",
        psm: int = 3,
        oem: int = 3,
        config_params: Optional[Dict[str, str]] = None,
        config: Optional[TesseractConfig] = None,  # Add config parameter for test compatibility
    ):
        """Initialize the OCR engine with configuration options.

        Args:
            tesseract_path: Path to the Tesseract executable. If None, uses the
                system default or the value from TESSDATA_PREFIX environment variable.
            languages: Language models to use, either as a string (comma-separated)
                or a list of language codes.
            psm: Page Segmentation Mode (default: 3 - fully automatic page segmentation).
                See Tesseract documentation for all options.
            oem: OCR Engine Mode (default: 3 - default, based on what is available).
                See Tesseract documentation for all options.
            config_params: Additional Tesseract configuration parameters.
            config: TesseractConfig object (for compatibility with tests).

        """
        # Handle TesseractConfig if provided
        if config:
            tesseract_path = config.tesseract_cmd
            languages = config.languages
            psm = config.psm.value if isinstance(config.psm, PageSegmentationMode) else config.psm
            oem = config.oem
            config_params = {}
            if config.custom_config:
                for param in config.custom_config.split():
                    if param.startswith("--"):
                        parts = param.split(" ", 1) if " " in param else [param, ""]
                        if len(parts) > 1:
                            config_params[parts[0].lstrip("-")] = parts[1]

        # Set Tesseract executable path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Convert language list to comma-separated string if needed
        if isinstance(languages, list):
            self.languages = "+".join(languages)
        else:
            self.languages = languages

        self.psm = psm
        self.oem = oem
        self.config_params = config_params or {}

        # For compatibility with existing tests
        self.config = config
        self.tesseract_cmd = tesseract_path

        # Build Tesseract config string
        self.config_string = f"--psm {psm} --oem {oem}"
        for key, value in self.config_params.items():
            self.config_string += f" -{key} {value}"

        logger.info(f"Initialized TesseractOCREngine with languages={self.languages}, psm={psm}, oem={oem}")

        # Verify Tesseract is available
        self._verify_tesseract_installation()

    def _verify_tesseract_installation(self):
        """Verify that Tesseract is properly installed and accessible."""
        try:
            pytesseract.get_tesseract_version()
            logger.debug("Tesseract installation verified successfully")
        except Exception as e:
            error_msg = f"Tesseract is not installed or not accessible. Error: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    # Alias for backward compatibility with tests
    _check_tesseract_installed = _verify_tesseract_installation

    def process_image(self, image: Image.Image) -> str:
        """Extract text from an image using Tesseract OCR.

        Args:
            image: PIL Image object to process.

        Returns:
            Extracted text as a string.

        Raises:
            DocumentProcessingError: If OCR processing fails.

        """
        try:
            logger.debug("Processing image with Tesseract OCR")
            text = pytesseract.image_to_string(image, lang=self.languages, config=self.config_string)

            logger.debug(f"OCR processing complete, extracted {len(text)} characters")
            return text

        except Exception as e:
            error_msg = f"OCR processing failed: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def process_multiple_images(self, images: List[Image.Image]) -> List[str]:
        """Process multiple images and return text for each.

        Args:
            images: List of PIL Image objects to process.

        Returns:
            List of extracted text strings, one per image.

        Raises:
            DocumentProcessingError: If OCR processing fails for any image.

        """
        results = []
        for i, image in enumerate(images):
            try:
                logger.debug(f"Processing image {i + 1} of {len(images)}")
                text = self.process_image(image)
                results.append(text)
            except Exception as e:
                error_msg = f"OCR processing failed for image {i + 1}: {str(e)}"
                logger.error(error_msg)
                raise DocumentProcessingError(error_msg) from e

        return results

    def get_supported_languages(self) -> List[str]:
        """Get list of languages supported by the Tesseract installation.

        Returns:
            List of language codes supported by Tesseract.

        Raises:
            ExternalServiceError: If Tesseract is not available or the command fails.

        """
        try:
            # Get the tesseract command
            tesseract_cmd = self.tesseract_cmd or pytesseract.pytesseract.tesseract_cmd or "tesseract"

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

    def _build_config_string(self):
        """Build the Tesseract configuration string."""
        return self.config_string
