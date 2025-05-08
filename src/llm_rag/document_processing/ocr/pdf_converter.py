"""PDF to image conversion for OCR processing.

This module provides functionality for converting PDF documents to high-resolution
images suitable for OCR processing. It uses PyMuPDF (fitz) to render PDF pages
and converts them to PIL Image objects with optional preprocessing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from llm_rag.utils.errors import DataAccessError, DocumentProcessingError, ErrorCode
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
        preprocessing_enabled: Whether to apply preprocessing to images (default: False).
        deskew_enabled: Whether to attempt to correct skewed text (default: False).
        threshold_enabled: Whether to apply adaptive thresholding (default: False).
        threshold_method: Thresholding method ('adaptive', 'otsu', or 'simple') (default: 'adaptive').
        contrast_adjust: Factor to adjust contrast (1.0 = no change) (default: 1.0).
        sharpen_enabled: Whether to apply sharpening to the image (default: False).
        denoise_enabled: Whether to apply denoising to the image (default: False).

    """

    dpi: int = 300
    output_format: str = 'png'
    use_alpha_channel: bool = False
    zoom_factor: float = 1.0
    first_page: Optional[int] = None
    last_page: Optional[int] = None
    preprocessing_enabled: bool = False
    deskew_enabled: bool = False
    threshold_enabled: bool = False
    threshold_method: str = 'adaptive'
    contrast_adjust: float = 1.0
    sharpen_enabled: bool = False
    denoise_enabled: bool = False


class PDFImageConverter:
    """Converts PDF pages to high-resolution images for OCR processing.

    This class takes a PDF file path and renders each page as a high-resolution
    image suitable for OCR processing, with optional preprocessing to improve OCR quality.
    """

    def __init__(self, dpi: int = 300, config: Optional[PDFImageConverterConfig] = None):
        """Initialize the PDF converter with configuration options.

        Args:
            dpi: The dots per inch (resolution) for rendering PDF pages.
                Higher values result in better quality but larger images.
            config: Complete configuration object. If provided, overrides the dpi parameter.

        """
        if config:
            self.config = config
        else:
            self.config = PDFImageConverterConfig(dpi=dpi)

        self.dpi = self.config.dpi

        if self.config.preprocessing_enabled and not OPENCV_AVAILABLE:
            logger.warning(
                'OpenCV is not available but preprocessing is enabled. '
                "Install it with 'pip install opencv-python' for preprocessing features."
            )
            self.config.preprocessing_enabled = False

        logger.info(
            f'Initialized PDFImageConverter with DPI: {self.dpi}, preprocessing: {self.config.preprocessing_enabled}'
        )

    def pdf_to_images(self, pdf_path: Union[str, Path]) -> Generator[Image.Image, None, None]:
        """Convert PDF pages to high-resolution images.

        Args:
            pdf_path: Path to the PDF file to convert.

        Returns:
            A generator yielding PIL Image objects for each page.

        Raises:
            DocumentProcessingError: If the file cannot be accessed or processed.

        """
        try:
            path = Path(pdf_path)
            logger.debug(f'Converting PDF to images: {path}')

            if not path.exists():
                raise FileNotFoundError(f'PDF file not found: {path}')

            # Open the PDF file
            doc = fitz.open(str(path))  # Convert path to string

            # Process each page
            for page_num, page in enumerate(doc):
                logger.debug(f'Processing page {page_num + 1} of {len(doc)}')

                # Set the rendering matrix for the desired DPI
                zoom = self.dpi / 72  # 72 DPI is the default PDF resolution
                matrix = fitz.Matrix(zoom, zoom)

                # Render page to pixmap
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)

                # Convert pixmap to PIL image
                image = Image.frombytes('RGB', [pixmap.width, pixmap.height], pixmap.samples)

                # Apply preprocessing if enabled
                if self.config.preprocessing_enabled:
                    image = self._preprocess_image(image)

                yield image

            doc.close()
            logger.info(f'Completed converting PDF {path} to images')

        except Exception as e:
            error_msg = f'Error processing PDF {pdf_path}: {str(e)}'
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def get_page_image(self, pdf_path: Union[str, Path], page_number: int) -> Optional[Image.Image]:
        """Get a specific page as an image from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            page_number: The 0-indexed page number to retrieve.

        Returns:
            PIL Image object of the requested page or None if the page doesn't exist.

        Raises:
            DocumentProcessingError: If the file cannot be accessed or processed.

        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            error_msg = f'PDF file not found: {pdf_path}'
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg)

        try:
            logger.info(f'Opening PDF file to extract page {page_number}: {pdf_path}')
            pdf_document = fitz.open(pdf_path)

            if page_number < 0 or page_number >= len(pdf_document):
                logger.warning(f'Page {page_number} out of range (0-{len(pdf_document) - 1})')
                return None

            page = pdf_document[page_number]

            # Set the rendering matrix for the desired DPI
            zoom = self.dpi / 72
            matrix = fitz.Matrix(zoom, zoom)

            # Render page to pixmap
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            # Convert pixmap to PIL image
            image = Image.frombytes('RGB', [pixmap.width, pixmap.height], pixmap.samples)

            # Apply preprocessing if enabled
            if self.config.preprocessing_enabled:
                image = self._preprocess_image(image)

            pdf_document.close()
            logger.info(f'Successfully extracted page {page_number} from {pdf_path}')

            return image

        except Exception as e:
            error_msg = f'Error extracting page {page_number} from PDF {pdf_path}: {str(e)}'
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

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
        logger.info(f'Converting PDF to images: {pdf_path}')
        doc = self._open_pdf_document(pdf_path)

        try:
            # Determine page range
            first_page = self.config.first_page - 1 if self.config.first_page else 0
            last_page = self.config.last_page if self.config.last_page else doc.page_count

            # Validate page range
            if first_page < 0 or first_page >= doc.page_count:
                raise ValueError(f'Invalid first_page {first_page + 1}. Valid range: 1-{doc.page_count}')
            if last_page > doc.page_count:
                raise ValueError(f'Invalid last_page {last_page}. Valid range: 1-{doc.page_count}')

            # Process each page in the specified range
            for page_num in range(first_page, last_page):
                logger.debug(f'Processing PDF page {page_num + 1}/{doc.page_count}')
                page = doc.load_page(page_num)
                image = self._render_page_to_image(page)
                yield page_num, image

        except ValueError as ve:
            # Re-raise ValueError as DataAccessError with appropriate error code
            raise DataAccessError(
                str(ve),
                error_code=ErrorCode.INVALID_INPUT,
                original_exception=ve,
                details={'pdf_path': str(pdf_path)},
            ) from ve
        finally:
            # Always close the document
            doc.close()
            logger.debug(f'Closed PDF document: {pdf_path}')

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
        logger.info(f'Converting PDF page {page_number} from {pdf_path}')
        doc = self._open_pdf_document(pdf_path)

        try:
            # Adjust for 0-indexing in PyMuPDF
            page_idx = page_number - 1

            # Validate page number
            if page_idx < 0 or page_idx >= doc.page_count:
                raise ValueError(f'Invalid page number {page_number}. Valid range: 1-{doc.page_count}')

            # Load and render the page
            page = doc.load_page(page_idx)
            return self._render_page_to_image(page)

        except ValueError as ve:
            # Re-raise ValueError as DataAccessError with appropriate error code
            raise DataAccessError(
                str(ve),
                error_code=ErrorCode.INVALID_INPUT,
                original_exception=ve,
                details={'pdf_path': str(pdf_path), 'page_number': page_number},
            ) from ve
        finally:
            # Always close the document
            doc.close()
            logger.debug(f'Closed PDF document: {pdf_path}')

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
        if doc is None:
            # If _open_pdf_document returns None, the error is already logged,
            # and the handle_exceptions decorator should handle reraise_for_testing
            raise DataAccessError(f'Failed to open PDF file: {pdf_path}', error_code=ErrorCode.FILE_NOT_FOUND)

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
        logger.info(f'Converting PDF pages from {pdf_path}')
        doc = self._open_pdf_document(pdf_path)
        images = []

        try:
            # If no specific pages are requested, convert all pages
            if page_numbers is None:
                page_numbers = range(doc.page_count)

            # Process each requested page
            for page_num in page_numbers:
                logger.debug(f'Processing PDF page {page_num + 1}/{doc.page_count}')
                page = doc.load_page(page_num)
                image = self._render_page_to_image(page, dpi=dpi)
                images.append(image)

            return images
        finally:
            # Always close the document
            doc.close()
            logger.debug(f'Closed PDF document: {pdf_path}')

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR quality.

        Args:
            image: The PIL image to preprocess.

        Returns:
            Preprocessed PIL image.

        """
        if not self.config.preprocessing_enabled:
            return image

        logger.debug('Applying image preprocessing')

        # Apply PIL-based preprocessing (always available)
        # Adjust contrast if needed
        if self.config.contrast_adjust != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.config.contrast_adjust)

        # Apply sharpening if enabled
        if self.config.sharpen_enabled:
            image = image.filter(ImageFilter.SHARPEN)

        # Skip OpenCV preprocessing if not available
        if not OPENCV_AVAILABLE:
            logger.debug('OpenCV not available, skipping advanced preprocessing')
            return image

        # Convert PIL to OpenCV format
        img = np.array(image)
        # Convert RGB to BGR (OpenCV format)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Apply grayscale conversion for preprocessing operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply denoising if enabled
        if self.config.denoise_enabled:
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Apply deskewing if enabled
        if self.config.deskew_enabled:
            gray = self._deskew_image(gray)

        # Apply thresholding if enabled
        if self.config.threshold_enabled:
            gray = self._apply_threshold(gray)

        # Convert back to RGB for PIL
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Convert back to PIL
        return Image.fromarray(processed)

    def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        """Correct skew in an image.

        Args:
            img: Grayscale OpenCV image.

        Returns:
            Deskewed grayscale OpenCV image.

        """
        if not OPENCV_AVAILABLE:
            return img

        logger.debug('Applying deskew operation')

        try:
            # Apply threshold to get a binary image
            thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Find all contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Find largest contour areas
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Use the largest contours only
            contours = contours[: min(10, len(contours))]

            # Compute the rotated bounding boxes
            angles = []
            for c in contours:
                # Minimum area rectangle
                rect = cv2.minAreaRect(c)
                # Get angle (adjusted for horizontal/vertical text)
                angle = rect[2]
                # Normalize angle between -45 and 45 degrees
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90

                # Only consider angles within a reasonable range
                if abs(angle) < 20:  # Ignore extreme angles
                    angles.append(angle)

            # If we have valid angles, compute the median
            if angles:
                skew_angle = np.median(angles)

                # Only correct if skew is significant
                if abs(skew_angle) > 0.5:
                    # Get image dimensions
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)

                    # Get rotation matrix
                    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

                    # Apply rotation
                    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated

            # If no valid angles or insignificant skew, return original
            return img
        except Exception as e:
            logger.warning(f'Deskew operation failed: {str(e)}')
            return img

    def _apply_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply thresholding to enhance text visibility.

        Args:
            img: Grayscale OpenCV image.

        Returns:
            Thresholded grayscale OpenCV image.

        """
        if not OPENCV_AVAILABLE:
            return img

        logger.debug(f'Applying {self.config.threshold_method} thresholding')

        try:
            if self.config.threshold_method == 'adaptive':
                # Apply adaptive thresholding
                # This works well for varying lighting conditions
                binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                return binary

            elif self.config.threshold_method == 'otsu':
                # Apply Otsu's thresholding
                # Good for bimodal images (clear foreground/background)
                _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return binary

            elif self.config.threshold_method == 'simple':
                # Apply simple binary thresholding
                _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
                return binary

            else:
                logger.warning(f'Unknown threshold method: {self.config.threshold_method}')
                return img

        except Exception as e:
            logger.warning(f'Thresholding operation failed: {str(e)}')
            return img
