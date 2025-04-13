"""PDF to image conversion for OCR processing.

This module provides functionality for converting PDF documents to high-resolution
images suitable for OCR processing. It uses PyMuPDF (fitz) to render PDF pages
and converts them to PIL Image objects.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import fitz  # PyMuPDF
from PIL import Image

from llm_rag.utils.errors import DataAccessError, DocumentProcessingError, ErrorCode, handle_exceptions
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PDFImageConverterConfig:
    """Configuration for PDF to image conversion.

    Attributes:
        dpi: Resolution for PDF rendering in dots per inch (default: 300).
        output_format: Image output format (default: 'png').
        use_alpha_channel: Whether to include alpha channel in output images (default: False).
        zoom_factor: Additional zoom factor applied to rendered pages (default: 1.0).
        first_page: First page to process, 1-indexed (default: None - start from the first page).
        last_page: Last page to process, 1-indexed (default: None - process until the last page).

    """

    dpi: int = 300
    output_format: str = "png"
    use_alpha_channel: bool = False
    zoom_factor: float = 1.0
    first_page: Optional[int] = None
    last_page: Optional[int] = None


class PDFImageConverter:
    """Converts PDF documents to high-resolution images for OCR processing.

    This class uses PyMuPDF to render PDF pages as high-quality images suitable
    for OCR processing. It can generate images for all pages or a specific range.
    """

    def __init__(self, config: Optional[PDFImageConverterConfig] = None):
        """Initialize the PDF to image converter.

        Args:
            config: Configuration options for PDF rendering. If None, default
                configuration will be used.

        """
        self.config = config or PDFImageConverterConfig()
        logger.info(f"Initialized PDFImageConverter with DPI={self.config.dpi}, format={self.config.output_format}")

    @handle_exceptions(
        error_type=DataAccessError,
        error_code=ErrorCode.FILE_NOT_FOUND,
        default_message="Failed to open PDF file",
    )
    def _open_pdf_document(self, pdf_path: Union[str, Path]) -> fitz.Document:
        """Open a PDF document using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A PyMuPDF Document object.

        Raises:
            DataAccessError: If the file cannot be opened or does not exist.

        """
        pdf_path_str = str(pdf_path)
        if not os.path.isfile(pdf_path_str):
            raise DataAccessError(
                f"PDF file not found: {pdf_path_str}",
                error_code=ErrorCode.FILE_NOT_FOUND,
            )

        try:
            return fitz.open(pdf_path_str)
        except Exception as e:
            raise DataAccessError(
                f"Failed to open PDF file: {pdf_path_str}",
                error_code=ErrorCode.INVALID_FILE_FORMAT,
                original_exception=e,
            ) from e

    @handle_exceptions(
        error_type=DocumentProcessingError,
        error_code=ErrorCode.DOCUMENT_PARSE_ERROR,
        default_message="Failed to convert PDF page to image",
    )
    def _render_page_to_image(self, page_or_path, page_num=0, dpi=None, alpha=None) -> Image.Image:
        """Render a single PDF page to a PIL Image.

        Args:
            page_or_path: Either a PyMuPDF Page object or a path to a PDF file.
            page_num: Page number to render (0-indexed). Only used if page_or_path is a file path.
            dpi: DPI for rendering (overrides config if provided).
            alpha: Whether to include alpha channel (overrides config if provided).

        Returns:
            PIL Image object of the rendered page.

        Raises:
            DocumentProcessingError: If page rendering fails.

        """
        # Set rendering parameters
        render_dpi = dpi if dpi is not None else self.config.dpi
        use_alpha = alpha if alpha is not None else self.config.use_alpha_channel

        # Calculate zoom matrix based on DPI
        zoom = render_dpi / 72 * self.config.zoom_factor  # 72 is the base DPI
        matrix = fitz.Matrix(zoom, zoom)

        try:
            # Handle different input types
            if isinstance(page_or_path, fitz.Page):
                page = page_or_path
            else:
                # Open the document and get the page
                doc = self._open_pdf_document(page_or_path)
                try:
                    page = doc.load_page(page_num)
                except Exception as e:
                    doc.close()
                    raise DocumentProcessingError(
                        f"Error loading page {page_num}",
                        error_code=ErrorCode.DOCUMENT_PARSE_ERROR,
                        original_exception=e,
                    ) from e

            # Render the page to a pixmap
            pix = page.get_pixmap(matrix=matrix, alpha=use_alpha)

            # Convert pixmap to PIL Image
            if use_alpha:
                img = Image.frombytes("RGBA", [pix.width, pix.height], pix.samples)
            else:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Close the document if we opened it here
            if not isinstance(page_or_path, fitz.Page) and "doc" in locals():
                doc.close()

            return img

        except Exception as e:
            page_num_info = getattr(page, "number", page_num) if "page" in locals() else page_num
            raise DocumentProcessingError(
                f"Error rendering page {page_num_info} to image",
                error_code=ErrorCode.DOCUMENT_PARSE_ERROR,
                original_exception=e,
                details={"page_number": page_num_info},
            ) from e

    def convert_pdf_to_images(self, pdf_path: Union[str, Path]) -> Generator[Tuple[int, Image.Image], None, None]:
        """Convert a PDF document to a sequence of images.

        Args:
            pdf_path: Path to the PDF file.

        Yields:
            Tuples of (page_number, PIL Image) for each page in the PDF.
            Page numbers are 0-indexed.

        Raises:
            DataAccessError: If the PDF file cannot be accessed or opened.
            DocumentProcessingError: If page rendering fails.

        """
        logger.info(f"Converting PDF to images: {pdf_path}")
        doc = self._open_pdf_document(pdf_path)

        try:
            # Determine page range
            first_page = self.config.first_page - 1 if self.config.first_page else 0
            last_page = self.config.last_page if self.config.last_page else doc.page_count

            # Validate page range
            if first_page < 0 or first_page >= doc.page_count:
                raise ValueError(f"Invalid first_page {first_page + 1}. Valid range: 1-{doc.page_count}")
            if last_page > doc.page_count:
                raise ValueError(f"Invalid last_page {last_page}. Valid range: 1-{doc.page_count}")

            # Process each page in the specified range
            for page_num in range(first_page, last_page):
                logger.debug(f"Processing PDF page {page_num + 1}/{doc.page_count}")
                page = doc.load_page(page_num)
                image = self._render_page_to_image(page)
                yield page_num, image

        except ValueError as ve:
            # Re-raise ValueError as DataAccessError with appropriate error code
            raise DataAccessError(
                str(ve),
                error_code=ErrorCode.INVALID_INPUT,
                original_exception=ve,
                details={"pdf_path": str(pdf_path)},
            ) from ve
        finally:
            # Always close the document
            doc.close()
            logger.debug(f"Closed PDF document: {pdf_path}")

    def convert_pdf_page_to_image(self, pdf_path: Union[str, Path], page_number: int) -> Image.Image:
        """Convert a specific PDF page to an image.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number to convert (1-indexed).

        Returns:
            PIL Image of the rendered page.

        Raises:
            DataAccessError: If the PDF file cannot be accessed or opened.
            DocumentProcessingError: If page rendering fails.
            ValueError: If the page number is invalid.

        """
        logger.info(f"Converting PDF page {page_number} from {pdf_path}")
        doc = self._open_pdf_document(pdf_path)

        try:
            # Adjust for 0-indexing in PyMuPDF
            page_idx = page_number - 1

            # Validate page number
            if page_idx < 0 or page_idx >= doc.page_count:
                raise ValueError(f"Invalid page number {page_number}. Valid range: 1-{doc.page_count}")

            # Load and render the page
            page = doc.load_page(page_idx)
            return self._render_page_to_image(page)

        except ValueError as ve:
            # Re-raise ValueError as DataAccessError with appropriate error code
            raise DataAccessError(
                str(ve),
                error_code=ErrorCode.INVALID_INPUT,
                original_exception=ve,
                details={"pdf_path": str(pdf_path), "page_number": page_number},
            ) from ve
        finally:
            # Always close the document
            doc.close()
            logger.debug(f"Closed PDF document: {pdf_path}")

    def get_pdf_page_count(self, pdf_path: Union[str, Path]) -> int:
        """Get the number of pages in a PDF document.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Number of pages in the PDF.

        Raises:
            DataAccessError: If the PDF file cannot be accessed or opened.

        """
        doc = self._open_pdf_document(pdf_path)
        try:
            return doc.page_count
        finally:
            doc.close()

    def convert_pdf_pages_to_images(self, pdf_path: Union[str, Path], page_numbers=None, dpi=None) -> list[Image.Image]:
        """Convert multiple PDF pages to images.

        Args:
            pdf_path: Path to the PDF file.
            page_numbers: List of page numbers to convert (0-indexed). If None, convert all pages.
            dpi: DPI for rendering (overrides config if provided).

        Returns:
            List of PIL Image objects.

        Raises:
            DataAccessError: If the PDF file cannot be accessed or opened.
            DocumentProcessingError: If page rendering fails.

        """
        logger.info(f"Converting PDF pages from {pdf_path}")
        doc = self._open_pdf_document(pdf_path)
        images = []

        try:
            # If no specific pages are requested, convert all pages
            if page_numbers is None:
                page_numbers = range(doc.page_count)

            # Process each requested page
            for page_num in page_numbers:
                logger.debug(f"Processing PDF page {page_num + 1}/{doc.page_count}")
                page = doc.load_page(page_num)
                image = self._render_page_to_image(page, dpi=dpi)
                images.append(image)

            return images
        finally:
            # Always close the document
            doc.close()
            logger.debug(f"Closed PDF document: {pdf_path}")
