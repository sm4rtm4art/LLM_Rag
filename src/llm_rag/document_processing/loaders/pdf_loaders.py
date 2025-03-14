"""PDF document loaders.

This module provides document loaders for PDF files using different backends.
"""

import logging
from pathlib import Path
from typing import Optional, Union

from ..processors import Documents
from .base import DocumentLoader, FileLoader, registry

logger = logging.getLogger(__name__)

# Optional imports for PDF processing
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. PDF loading capabilities will be limited.")

try:
    from PyPDF2 import PdfReader

    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available. Some PDF loading capabilities may be affected.")


class PDFLoader(DocumentLoader, FileLoader):
    """Load documents from PDF files.

    This loader extracts text content from PDF files. It tries to use PyMuPDF (fitz)
    first, and falls back to PyPDF2 if PyMuPDF is not available.
    """

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        extract_images: bool = False,
        extract_tables: bool = False,
        page_separator: str = "\n\n",
        start_page: int = 0,
        end_page: Optional[int] = None,
    ):
        r"""Initialize the PDF loader.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], optional
            Path to the PDF file, by default None
        extract_images : bool, optional
            Whether to extract images from the PDF, by default False
        extract_tables : bool, optional
            Whether to extract tables from the PDF, by default False
        page_separator : str, optional
            Separator to use between pages, by default "\n\n"
        start_page : int, optional
            Page to start extraction from (0-indexed), by default 0
        end_page : Optional[int], optional
            Page to end extraction at (0-indexed, inclusive), by default None (all pages)

        """
        self.file_path = Path(file_path) if file_path else None
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.page_separator = page_separator
        self.start_page = start_page
        self.end_page = end_page

    def load(self) -> Documents:
        """Load documents from the PDF file specified during initialization.

        Returns
        -------
        Documents
            List of documents, one per page of the PDF.

        Raises
        ------
        ValueError
            If file_path was not provided during initialization.

        """
        if not self.file_path:
            raise ValueError("No file path provided. Either initialize with a file path or use load_from_file.")

        return self.load_from_file(self.file_path)

    def load_from_file(self, file_path: Union[str, Path]) -> Documents:
        """Load documents from a PDF file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the PDF file to load.

        Returns
        -------
        Documents
            List of documents, one per page of the PDF.

        Raises
        ------
        Exception
            If PDF loading fails.

        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            # Try to use PyMuPDF first
            if PYMUPDF_AVAILABLE:
                return self._load_with_pymupdf(file_path)
            # Fall back to PyPDF2
            elif PYPDF2_AVAILABLE:
                return self._load_with_pypdf2(file_path)
            else:
                raise ImportError("No PDF processing library available. Install PyMuPDF or PyPDF2.")
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            raise Exception(f"Error loading PDF file {file_path}: {str(e)}") from e

    def _load_with_pymupdf(self, file_path: Path) -> Documents:
        """Load PDF using PyMuPDF.

        Parameters
        ----------
        file_path : Path
            Path to the PDF file.

        Returns
        -------
        Documents
            List of documents, one per page.

        """
        documents = []

        with fitz.open(str(file_path)) as pdf:
            # Determine end page if not specified
            end_page = self.end_page if self.end_page is not None else len(pdf) - 1

            # Validate page range
            start_page = max(0, min(self.start_page, len(pdf) - 1))
            end_page = max(start_page, min(end_page, len(pdf) - 1))

            # Extract text from each page
            for page_num in range(start_page, end_page + 1):
                page = pdf[page_num]

                # Extract text content
                text = page.get_text()

                # Create metadata
                metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "filetype": "pdf",
                    "page": page_num,
                    "total_pages": len(pdf),
                }

                # Add document for this page
                documents.append({"content": text, "metadata": metadata})

            # TODO: Implement image extraction if self.extract_images is True
            # TODO: Implement table extraction if self.extract_tables is True

        return documents

    def _load_with_pypdf2(self, file_path: Path) -> Documents:
        """Load PDF using PyPDF2.

        Parameters
        ----------
        file_path : Path
            Path to the PDF file.

        Returns
        -------
        Documents
            List of documents, one per page.

        """
        documents = []

        with open(file_path, "rb") as file:
            pdf = PdfReader(file)

            # Determine end page if not specified
            end_page = self.end_page if self.end_page is not None else len(pdf.pages) - 1

            # Validate page range
            start_page = max(0, min(self.start_page, len(pdf.pages) - 1))
            end_page = max(start_page, min(end_page, len(pdf.pages) - 1))

            # Extract text from each page
            for page_num in range(start_page, end_page + 1):
                page = pdf.pages[page_num]

                # Extract text content
                text = page.extract_text()

                # Create metadata
                metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "filetype": "pdf",
                    "page": page_num,
                    "total_pages": len(pdf.pages),
                }

                # Add document for this page
                if text:
                    documents.append({"content": text, "metadata": metadata})

        return documents


class EnhancedPDFLoader(PDFLoader):
    """Enhanced PDF loader with support for images, tables, and OCR.

    This loader provides more advanced PDF processing capabilities, including
    image extraction, table detection, and optical character recognition (OCR).
    """

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        extract_images: bool = True,
        extract_tables: bool = True,
        use_ocr: bool = False,
        ocr_languages: str = "eng",
        page_separator: str = "\n\n",
        start_page: int = 0,
        end_page: Optional[int] = None,
    ):
        r"""Initialize the enhanced PDF loader.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], optional
            Path to the PDF file, by default None
        extract_images : bool, optional
            Whether to extract images from the PDF, by default True
        extract_tables : bool, optional
            Whether to extract tables from the PDF, by default True
        use_ocr : bool, optional
            Whether to use OCR on image content, by default False
        ocr_languages : str, optional
            Languages to use for OCR, by default "eng"
        page_separator : str, optional
            Separator to use between pages, by default "\n\n"
        start_page : int, optional
            Page to start extraction from (0-indexed), by default 0
        end_page : Optional[int], optional
            Page to end extraction at (0-indexed, inclusive), by default None (all pages)

        """
        super().__init__(
            file_path=file_path,
            extract_images=extract_images,
            extract_tables=extract_tables,
            page_separator=page_separator,
            start_page=start_page,
            end_page=end_page,
        )
        self.use_ocr = use_ocr
        self.ocr_languages = ocr_languages

        # Check for optional dependencies
        if self.use_ocr:
            try:
                import importlib.util

                has_pil = importlib.util.find_spec("PIL") is not None
                has_pytesseract = importlib.util.find_spec("pytesseract") is not None

                if has_pil and has_pytesseract:
                    self._has_ocr = True
                else:
                    missing = []
                    if not has_pil:
                        missing.append("PIL")
                    if not has_pytesseract:
                        missing.append("pytesseract")
                    raise ImportError(f"Missing required modules: {', '.join(missing)}")
            except ImportError as e:
                logger.warning(f"OCR requested but dependencies not available: {e}")
                self._has_ocr = False
        else:
            self._has_ocr = False

        if self.extract_tables:
            try:
                import importlib.util

                if importlib.util.find_spec("tabula") is not None:
                    self._has_tabula = True
                else:
                    raise ImportError("tabula module not found")
            except ImportError:
                logger.warning("Table extraction requested but tabula-py not available.")
                self._has_tabula = False
        else:
            self._has_tabula = False

    def _load_with_pymupdf(self, file_path: Path) -> Documents:
        """Override the PyMuPDF loading to add enhanced features.

        Parameters
        ----------
        file_path : Path
            Path to the PDF file.

        Returns
        -------
        Documents
            List of documents with enhanced content.

        """
        # First use the basic PDF loading
        documents = super()._load_with_pymupdf(file_path)

        # Then add enhanced features if requested and available
        if (self.extract_images or self.extract_tables) and len(documents) > 0:
            # TODO: Implement enhanced features:
            # - Image extraction
            # - Table extraction
            # - OCR processing
            pass

        return documents


# Register the loaders
registry.register(PDFLoader, extensions=["pdf"])
registry.register(EnhancedPDFLoader, name="EnhancedPDFLoader", extensions=[])
