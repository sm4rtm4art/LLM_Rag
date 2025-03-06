"""Document loaders for the RAG system."""

import csv
import io
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self) -> List[Dict[str, Union[str, Dict]]]:
        """Load documents from a source.

        Returns
        -------
            List of documents, where each document is a dictionary with
            'content' and 'metadata' keys.

        """
        pass


class TextFileLoader(DocumentLoader):
    """Load documents from a text file."""

    def __init__(self, file_path: Union[str, Path]):
        """Initialize the loader.

        Args:
        ----
            file_path: Path to the text file.

        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def load(self) -> List[Dict[str, Union[str, Dict]]]:
        """Load the text file and return it as a document.

        Returns
        -------
            List containing a single document with the file content
            and metadata.

        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        metadata = {
            "source": str(self.file_path),
            "filename": self.file_path.name,
            "filetype": "txt",
        }

        return [{"content": content, "metadata": metadata}]


class PDFLoader(DocumentLoader):
    """Load documents from a PDF file."""

    def __init__(
        self,
        file_path: Union[str, Path],
        extract_images: bool = False,
        extract_tables: bool = False,
        use_enhanced_extraction: bool = False,
        output_dir: Optional[str] = None,
    ):
        """Initialize the loader.

        Args:
        ----
            file_path: Path to the PDF file.
            extract_images: Whether to extract images from the PDF.
            extract_tables: Whether to extract tables from the PDF.
            use_enhanced_extraction: Whether to use the enhanced PDF extraction.
            output_dir: Directory to save extracted tables and images.

        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.use_enhanced_extraction = use_enhanced_extraction
        self.output_dir = output_dir

        # Check if PyPDF2 is installed
        try:
            import PyPDF2  # noqa: F401
        except ImportError as err:
            raise ImportError("PyPDF2 is required for PDF loading. Install it with 'pip install PyPDF2'") from err

    def load(self) -> List[Dict[str, Union[str, Dict]]]:
        """Load the PDF file.

        Returns
        -------
            List containing a single document with the PDF content.

        """
        # If enhanced extraction is enabled, use the EnhancedPDFProcessor
        if self.use_enhanced_extraction:
            return self._load_enhanced()

        # Otherwise, use the standard extraction
        import PyPDF2

        documents = []

        with open(self.file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)

            # Extract text from each page
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"

        # Create the main document with text content
        metadata = {
            "source": str(self.file_path),
            "filename": self.file_path.name,
            "filetype": "pdf",
            "pages": len(pdf_reader.pages),
        }

        documents.append({"content": text, "metadata": metadata})

        # Extract tables if requested
        if self.extract_tables:
            try:
                tables = self._extract_tables()
                for i, table in enumerate(tables):
                    table_metadata = metadata.copy()
                    table_metadata["content_type"] = "table"
                    table_metadata["table_index"] = i

                    documents.append({"content": table, "metadata": table_metadata})

                    logger.info(f"Extracted table {i} from {self.file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to extract tables from {self.file_path.name}: {e}")

        # Extract images if requested
        if self.extract_images:
            try:
                images = self._extract_images()
                for i, (image_path, image_text) in enumerate(images):
                    image_metadata = metadata.copy()
                    image_metadata["content_type"] = "image"
                    image_metadata["image_index"] = i
                    image_metadata["image_path"] = str(image_path)

                    documents.append({"content": image_text, "metadata": image_metadata})

                    logger.info(f"Extracted image {i} from {self.file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to extract images from {self.file_path.name}: {e}")

        return documents

    def _load_enhanced(self) -> List[Dict[str, Union[str, Dict]]]:
        """Load the PDF file using the enhanced PDF processor.

        Returns
        -------
            List of documents with the PDF content, tables, and images.

        """
        try:
            # Import the EnhancedPDFProcessor
            try:
                import importlib.util
                import os
                import sys

                # Log Python path for debugging
                logger.info(f"Python path: {sys.path}")
                logger.info(f"Python executable: {sys.executable}")
                logger.info(f"Current directory: {os.getcwd()}")

                # Add the project root to the Python path if not already there
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                    logger.info(f"Added {project_root} to Python path")

                # Try direct import using importlib
                module_path = os.path.join(project_root, "scripts", "analytics", "rag_integration.py")
                if os.path.exists(module_path):
                    spec = importlib.util.spec_from_file_location("rag_integration", module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        EnhancedPDFProcessor = module.EnhancedPDFProcessor
                        logger.info("Successfully imported EnhancedPDFProcessor")
                    else:
                        raise ImportError("Failed to load module spec")
                else:
                    raise ImportError(f"Module path not found: {module_path}")

            except (ImportError, AttributeError) as e:
                logger.error(f"Error in enhanced PDF extraction: {e}")
                logger.warning("Falling back to standard extraction.")
                return self.load()

            # Create the processor
            processor = EnhancedPDFProcessor(
                output_dir=self.output_dir,
                save_tables=self.extract_tables,
                save_images=self.extract_images,
                verbose=True,
            )

            # Process the PDF
            result = processor.process_pdf(str(self.file_path))

            # Get the documents
            if "documents" in result and result["documents"]:
                return result["documents"]
            else:
                logger.warning(f"No documents found in enhanced extraction for {self.file_path.name}")
                return self.load()

        except Exception as e:
            logger.error(f"Error in enhanced PDF extraction: {e}")
            logger.warning("Falling back to standard extraction.")
            return self.load()

    def _extract_tables(self) -> List[str]:
        """Extract tables from the PDF.

        Returns
        -------
            List of table contents as strings.

        """
        # Debug information about Python path and environment
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Python executable: {sys.executable}")

        try:
            logger.info("Attempting to import tabula-py...")
            import tabula

            logger.info(
                f"Successfully imported tabula-py: "
                f"{tabula.__version__ if hasattr(tabula, '__version__') else 'unknown version'}"
            )
        except ImportError as e:
            logger.warning(f"tabula-py import error: {e}")
            logger.warning("tabula-py is required for table extraction. Install it with 'pip install tabula-py'")
            return []
        except Exception as e:
            logger.warning(f"Unexpected error importing tabula-py: {e}")
            return []

        try:
            # Extract tables from the PDF
            logger.info(f"Extracting tables from {self.file_path}")
            tables = tabula.read_pdf(str(self.file_path), pages="all", multiple_tables=True)
            logger.info(f"Found {len(tables)} tables")

            # Convert tables to string representation
            table_strings = []
            for table in tables:
                # Convert DataFrame to CSV string
                csv_buffer = io.StringIO()
                table.to_csv(csv_buffer)
                table_strings.append(csv_buffer.getvalue())

            return table_strings
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")
            return []

    def _extract_images(self) -> List[tuple[Path, str]]:
        """Extract images from the PDF and save them to disk.

        Returns
        -------
            List of tuples containing (image_path, image_description).

        """
        # Debug information about Python path and environment
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Python executable: {sys.executable}")

        try:
            logger.info("Attempting to import pdf2image and pytesseract...")
            import pytesseract
            from pdf2image import convert_from_path

            logger.info("Successfully imported pdf2image and pytesseract")
            logger.info(
                f"pdf2image version: "
                f"{convert_from_path.__module__ if hasattr(convert_from_path, '__module__') else 'unknown'}"
            )
            logger.info(
                f"pytesseract version: {pytesseract.__version__ if hasattr(pytesseract, '__version__') else 'unknown'}"
            )
        except ImportError as e:
            logger.warning(f"Image extraction import error: {e}")
            logger.warning(
                "pdf2image, pytesseract, and Pillow are required for image extraction. "
                "Install them with 'pip install pdf2image pytesseract Pillow'"
            )
            return []
        except Exception as e:
            logger.warning(f"Unexpected error importing image extraction libraries: {e}")
            return []

        try:
            # Create directory for images
            output_dir = Path(f"{self.file_path.stem}_images")
            output_dir.mkdir(exist_ok=True)

            # Convert PDF pages to images
            logger.info(f"Converting PDF to images: {self.file_path}")
            images = convert_from_path(str(self.file_path))
            logger.info(f"Converted {len(images)} pages to images")

            result = []
            for i, image in enumerate(images):
                # Save the image
                image_path = output_dir / f"page{i + 1}.png"
                image.save(str(image_path), "PNG")

                # Extract text from image using OCR
                try:
                    logger.info(f"Performing OCR on image {i + 1}")
                    image_text = pytesseract.image_to_string(image)
                    logger.info(f"OCR completed for image {i + 1}")
                except Exception as e:
                    logger.warning(f"OCR failed for image {i + 1}: {e}")
                    image_text = f"[Image from page {i + 1}]"

                result.append((image_path, image_text))

            return result
        except Exception as e:
            logger.warning(f"Error extracting images: {e}")
            return []


class CSVLoader(DocumentLoader):
    """Load documents from a CSV file."""

    def __init__(
        self,
        file_path: Union[str, Path],
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        """Initialize the loader.

        Args:
        ----
            file_path: Path to the CSV file.
            content_columns: List of column names to include in the content.
                If None, all columns are included.
            metadata_columns: List of column names to include in the metadata.
                If None, no columns are included in metadata.

        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.content_columns = content_columns
        self.metadata_columns = metadata_columns

    def load(self) -> List[Dict[str, Union[str, Dict]]]:
        """Load the CSV file.

        Returns
        -------
            List of documents, one per row in the CSV file.

        """
        documents = []

        with open(self.file_path, "r", encoding="utf-8", newline="") as f:
            csv_reader = csv.DictReader(f)

            # Get all column names
            all_columns = csv_reader.fieldnames or []

            # Determine which columns to use for content
            content_columns = self.content_columns or all_columns

            for row in csv_reader:
                # Build content string
                content_parts = []
                for col in content_columns:
                    # Check if column exists and has value
                    if col in row and row[col]:
                        content_parts.append(f"{col}: {row[col]}")

                content = "\n".join(content_parts)

                # Build metadata
                metadata = {
                    "source": str(self.file_path),
                    "filename": self.file_path.name,
                    "filetype": "csv",
                }

                # Add specified columns to metadata
                if self.metadata_columns:
                    for col in self.metadata_columns:
                        if col in row:
                            metadata[col] = row[col]

                documents.append({"content": content, "metadata": metadata})

        return documents


class DirectoryLoader(DocumentLoader):
    """Load documents from a directory."""

    def __init__(
        self,
        directory_path: Union[str, Path],
        recursive: bool = False,
        glob_pattern: Optional[str] = None,
        extract_images: bool = False,
        extract_tables: bool = False,
        use_enhanced_extraction: bool = False,
        output_dir: Optional[str] = None,
    ):
        """Initialize the loader.

        Args:
        ----
            directory_path: Path to the directory.
            recursive: Whether to recursively search for files in subdirectories.
            glob_pattern: Pattern to match files against.
            extract_images: Whether to extract images from PDFs.
            extract_tables: Whether to extract tables from PDFs.
            use_enhanced_extraction: Whether to use the enhanced PDF extraction.
            output_dir: Directory to save extracted tables and images.

        """
        self.directory_path = Path(directory_path)
        if not self.directory_path.exists():
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        if not self.directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        self.recursive = recursive
        self.glob_pattern = glob_pattern
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.use_enhanced_extraction = use_enhanced_extraction
        self.output_dir = output_dir

    def load(self) -> List[Dict[str, Union[str, Dict]]]:
        """Load documents from the directory.

        Returns
        -------
            List of documents.

        """
        documents = []

        # Get all files in the directory
        if self.glob_pattern:
            if self.recursive:
                files = list(self.directory_path.glob(f"**/{self.glob_pattern}"))
            else:
                files = list(self.directory_path.glob(self.glob_pattern))
        else:
            if self.recursive:
                files = [f for f in self.directory_path.glob("**/*") if f.is_file()]
            else:
                files = [f for f in self.directory_path.glob("*") if f.is_file()]

        # Process each file
        for file_path in files:
            try:
                # Choose the appropriate loader based on file extension
                if file_path.suffix.lower() == ".pdf":
                    loader = PDFLoader(
                        file_path,
                        extract_images=self.extract_images,
                        extract_tables=self.extract_tables,
                        use_enhanced_extraction=self.use_enhanced_extraction,
                        output_dir=self.output_dir,
                    )
                elif file_path.suffix.lower() == ".csv":
                    loader = CSVLoader(file_path)
                else:
                    # Default to text loader for other file types
                    loader = TextFileLoader(file_path)

                # Load the document
                file_documents = loader.load()
                documents.extend(file_documents)
                logger.info(f"Loaded {len(file_documents)} documents from {file_path}")
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                print(f"Error loading {file_path}: {e}")

        return documents
