"""Module for orchestrating the OCR pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from llm_rag.document_processing.ocr.ocr_engine import TesseractOCREngine
from llm_rag.document_processing.ocr.pdf_converter import PDFImageConverter
from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OCRPipelineConfig:
    """Configuration settings for OCR pipeline.

    This class encapsulates all configuration options for the OCR pipeline,
    including PDF conversion settings and OCR engine settings.
    """

    # PDF Converter Settings
    pdf_dpi: int = 300

    # OCR Engine Settings
    tesseract_path: Optional[str] = None
    languages: Union[str, List[str]] = "eng"
    psm: int = 3  # Page Segmentation Mode
    oem: int = 3  # OCR Engine Mode
    config_params: Optional[Dict[str, str]] = None

    # Pipeline Settings
    process_pages: Optional[List[int]] = None  # Pages to process (None = all)
    parallelize: bool = False  # Future enhancement for parallel processing


class OCRPipeline:
    """Pipeline for converting PDFs to text using OCR.

    This class orchestrates the complete OCR workflow:
    1. Converting PDF pages to high-quality images
    2. Processing those images with OCR to extract text

    The pipeline can be configured via OCRPipelineConfig to customize
    both the PDF conversion and OCR processing steps.
    """

    def __init__(self, config: Optional[OCRPipelineConfig] = None):
        """Initialize the OCR pipeline with configuration options.

        Args:
            config: Configuration for the pipeline. If None, default
                configuration will be used.

        """
        self.config = config or OCRPipelineConfig()
        logger.info("Initializing OCR pipeline")

        # Initialize the PDF converter
        self.pdf_converter = PDFImageConverter(dpi=self.config.pdf_dpi)

        # Initialize the OCR engine
        self.ocr_engine = TesseractOCREngine(
            tesseract_path=self.config.tesseract_path,
            languages=self.config.languages,
            psm=self.config.psm,
            oem=self.config.oem,
            config_params=self.config.config_params,
        )

        logger.info("OCR pipeline initialized successfully")

    def process_pdf(self, pdf_path: Union[str, Path]) -> str:
        """Process an entire PDF document and return the OCR text.

        Args:
            pdf_path: Path to the PDF file to process.

        Returns:
            Extracted text from all processed pages combined into a single string.

        Raises:
            DocumentProcessingError: If processing fails at any stage.

        """
        try:
            pdf_path = Path(pdf_path)
            logger.info(f"Processing PDF: {pdf_path}")

            # Convert PDF pages to images
            if self.config.process_pages is not None:
                # Process specific pages
                images = []
                for page_num in self.config.process_pages:
                    image = self.pdf_converter.get_page_image(pdf_path, page_num)
                    if image:
                        images.append(image)
            else:
                # Process all pages
                images = list(self.pdf_converter.pdf_to_images(pdf_path))

            if not images:
                error_msg = f"No valid images extracted from PDF: {pdf_path}"
                logger.error(error_msg)
                raise DocumentProcessingError(error_msg)

            logger.info(f"Extracted {len(images)} images from PDF")

            # Process images with OCR
            text_pages = self.ocr_engine.process_multiple_images(images)

            # Combine text from all pages
            full_text = "\n\n".join(text_pages)

            logger.info(f"OCR completed for {pdf_path}, extracted {len(full_text)} characters")

            return full_text

        except Exception as e:
            error_msg = f"OCR pipeline failed for PDF {pdf_path}: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def process_pdf_pages(self, pdf_path: Union[str, Path], page_numbers: List[int]) -> Dict[int, str]:
        """Process specific pages of a PDF document.

        Args:
            pdf_path: Path to the PDF file to process.
            page_numbers: List of page numbers to process (0-indexed).

        Returns:
            Dictionary mapping page numbers to their extracted text.

        Raises:
            DocumentProcessingError: If processing fails at any stage.

        """
        try:
            pdf_path = Path(pdf_path)
            logger.info(f"Processing specific pages from PDF: {pdf_path}")

            results = {}
            for page_num in page_numbers:
                logger.info(f"Processing page {page_num}")

                # Convert PDF page to image
                image = self.pdf_converter.get_page_image(pdf_path, page_num)

                if image:
                    # Process image with OCR
                    text = self.ocr_engine.process_image(image)
                    results[page_num] = text
                    logger.debug(f"Completed OCR for page {page_num}")
                else:
                    logger.warning(f"Could not extract image for page {page_num}")

            if not results:
                error_msg = f"No text extracted from specified pages in PDF: {pdf_path}"
                logger.error(error_msg)
                raise DocumentProcessingError(error_msg)

            logger.info(f"OCR completed for {len(results)} pages from {pdf_path}")
            return results

        except Exception as e:
            error_msg = f"OCR pipeline failed for PDF {pdf_path}: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e
