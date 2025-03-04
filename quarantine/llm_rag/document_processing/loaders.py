"""Document loaders for the RAG system."""

import csv
import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

# Define a type variable for document loaders
T = TypeVar("T", bound="DocumentLoader")


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
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

    def load(self) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """Load the text file.

        Returns
        -------
            List containing a single document with the file content.

        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        metadata = {
            "source": str(self.file_path),
            "filename": self.file_path.name,
            "filetype": self.file_path.suffix.lstrip(".") or "text",
        }

        return [{"content": content, "metadata": metadata}]


class PDFLoader(DocumentLoader):
    """Load documents from a PDF file."""

    def __init__(self, file_path: Union[str, Path]):
        """Initialize the loader.

        Args:
        ----
            file_path: Path to the PDF file.

        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if PyPDF2 is installed
        try:
            import PyPDF2  # noqa: F401
        except ImportError as err:
            raise ImportError("PyPDF2 is required for PDF loading. Install it with 'pip install PyPDF2'") from err

    def load(self) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """Load the PDF file.

        Returns
        -------
            List containing a single document with the PDF content.

        """
        import PyPDF2

        with open(self.file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)

            # Extract text from each page
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"

        metadata = {
            "source": str(self.file_path),
            "filename": self.file_path.name,
            "filetype": "pdf",
            "pages": len(pdf_reader.pages),
        }

        return [{"content": text, "metadata": metadata}]


class EnhancedPDFLoader(DocumentLoader):
    """Enhanced PDF loader with table and image extraction capabilities."""

    def __init__(
        self,
        file_path: Union[str, Path],
        extract_tables: bool = True,
        extract_images: bool = True,
        image_output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the enhanced PDF loader.

        Args:
        ----
            file_path: Path to the PDF file.
            extract_tables: Whether to extract tables from the PDF.
            extract_images: Whether to extract images from the PDF.
            image_output_dir: Directory to save extracted images.
                If None, images will be saved in a subdirectory of the PDF location.

        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.extract_tables = extract_tables
        self.extract_images = extract_images

        # Set up image output directory
        if image_output_dir:
            self.image_output_dir = Path(image_output_dir)
        else:
            # Create a directory for images based on the PDF filename
            pdf_dir = self.file_path.parent
            pdf_name = self.file_path.stem
            self.image_output_dir = pdf_dir / f"{pdf_name}_images"

        # Create the image directory if it doesn't exist and we're extracting images
        if self.extract_images and not self.image_output_dir.exists():
            os.makedirs(self.image_output_dir, exist_ok=True)

        # Check for required dependencies
        try:
            import PyPDF2  # noqa: F401
        except ImportError as err:
            raise ImportError("PyPDF2 is required for PDF loading. Install it with 'pip install PyPDF2'") from err

        if self.extract_tables:
            try:
                import tabula  # noqa: F401
            except ImportError as err:
                raise ImportError(
                    "tabula-py is required for table extraction. Install it with 'pip install tabula-py'"
                ) from err

        if self.extract_images:
            try:
                import fitz  # noqa: F401
            except ImportError as err:
                raise ImportError(
                    "PyMuPDF is required for image extraction. Install it with 'pip install pymupdf'"
                ) from err

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF.

        Args:
        ----
            pdf_path: Path to the PDF file.

        Returns:
        -------
            Extracted text content.

        """
        import PyPDF2

        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"

        return text

    def _extract_tables(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables from PDF.

        Args:
        ----
            pdf_path: Path to the PDF file.

        Returns:
        -------
            List of extracted tables with metadata.

        """
        import tabula

        tables_data = []

        # Extract all tables from the PDF
        try:
            # Use tabula to extract tables
            dfs = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True)

            for i, df in enumerate(dfs):
                if not df.empty:
                    # Convert DataFrame to string representation
                    table_str = df.to_string(index=False)

                    # Create a CSV string representation
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_str = csv_buffer.getvalue()

                    # Add table data with metadata
                    tables_data.append(
                        {
                            "content": table_str,
                            "metadata": {
                                "source": str(pdf_path),
                                "filename": pdf_path.name,
                                "filetype": "table",
                                "table_index": i,
                                "table_format": "string",
                                "csv_content": csv_str,
                                "columns": df.columns.tolist(),
                                "rows": len(df),
                            },
                        }
                    )
        except Exception as e:
            print(f"Error extracting tables from {pdf_path}: {str(e)}")

        return tables_data

    def _extract_images(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract images from PDF.

        Args:
        ----
            pdf_path: Path to the PDF file.

        Returns:
        -------
            List of image references with metadata.

        """
        import fitz  # PyMuPDF

        images_data = []

        try:
            # Open the PDF
            pdf_document = fitz.open(str(pdf_path))

            # Iterate through pages
            for page_num, page in enumerate(pdf_document):
                # Get images from the page
                image_list = page.get_images(full=True)

                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]  # Image reference

                    # Extract the image
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Generate a filename for the image
                    image_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                    image_path = self.image_output_dir / image_filename

                    # Save the image
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)

                    # Get image dimensions and other metadata
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Create a caption based on page number and image index
                    caption = f"Figure from page {page_num + 1}, image {img_index + 1}"

                    # Add image data with metadata
                    images_data.append(
                        {
                            "content": caption,
                            "metadata": {
                                "source": str(pdf_path),
                                "filename": pdf_path.name,
                                "filetype": "image",
                                "image_path": str(image_path),
                                "image_filename": image_filename,
                                "page_num": page_num + 1,
                                "image_index": img_index + 1,
                                "width": width,
                                "height": height,
                                "format": image_ext,
                            },
                        }
                    )

            pdf_document.close()

        except Exception as e:
            print(f"Error extracting images from {pdf_path}: {str(e)}")

        return images_data

    def load(self) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """Load the PDF file with enhanced extraction.

        Returns
        -------
            List of documents including text content, tables, and image references.

        """
        documents = []

        # Extract text content
        text_content = self._extract_text(self.file_path)

        # Add text document
        documents.append(
            {
                "content": text_content,
                "metadata": {
                    "source": str(self.file_path),
                    "filename": self.file_path.name,
                    "filetype": "pdf_text",
                },
            }
        )

        # Extract and add tables if requested
        if self.extract_tables:
            table_documents = self._extract_tables(self.file_path)
            documents.extend(table_documents)

        # Extract images if requested
        image_documents = []
        if self.extract_images:
            image_documents = self._extract_images(self.file_path)
            documents.extend(image_documents)

        return documents


class DINStandardLoader(EnhancedPDFLoader):
    """Specialized loader for DIN standard documents with structured extraction."""

    def __init__(
        self,
        file_path: Union[str, Path],
        extract_tables: bool = True,
        extract_images: bool = True,
        image_output_dir: Optional[Union[str, Path]] = None,
        extract_drawings: bool = True,
    ):
        """Initialize the DIN standard loader.

        Args:
        ----
            file_path: Path to the DIN standard PDF file.
            extract_tables: Whether to extract tables from the PDF.
            extract_images: Whether to extract images from the PDF.
            image_output_dir: Directory to save extracted images.
            extract_drawings: Whether to specifically identify and extract technical drawings.

        """
        super().__init__(file_path, extract_tables, extract_images, image_output_dir)
        self.extract_drawings = extract_drawings

    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify and extract sections from DIN standard text.

        Args:
        ----
            text: The full text content of the DIN standard.

        Returns:
        -------
            List of sections with their content and metadata.

        """
        import re

        # Common section patterns in DIN standards
        section_patterns = [
            r"(\d+\s+Scope\s*\n)",
            r"(\d+\s+Normative references\s*\n)",
            r"(\d+\s+Terms and definitions\s*\n)",
            r"(\d+\s+Requirements\s*\n)",
            r"(\d+\s+Testing\s*\n)",
            r"(\d+\s+Marking\s*\n)",
            r"(\d+\s+Annex\s+[A-Z])",
        ]

        # Combine patterns
        combined_pattern = "|".join(section_patterns)

        # Find all section headers
        section_matches = list(re.finditer(combined_pattern, text, re.IGNORECASE))

        sections = []

        # Process each section
        for i, match in enumerate(section_matches):
            section_start = match.start()
            section_title = match.group().strip()

            # Determine section end (start of next section or end of text)
            if i < len(section_matches) - 1:
                section_end = section_matches[i + 1].start()
            else:
                section_end = len(text)

            # Extract section content
            section_content = text[section_start:section_end].strip()

            # Add section with metadata
            sections.append(
                {
                    "content": section_content,
                    "metadata": {
                        "source": str(self.file_path),
                        "filename": self.file_path.name,
                        "filetype": "din_section",
                        "section_title": section_title,
                        "section_index": i,
                    },
                }
            )

        # If no sections were found, return the whole text as one section
        if not sections:
            sections.append(
                {
                    "content": text,
                    "metadata": {
                        "source": str(self.file_path),
                        "filename": self.file_path.name,
                        "filetype": "din_section",
                        "section_title": "Full Document",
                        "section_index": 0,
                    },
                }
            )

        return sections

    def _identify_drawings(self, image_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify technical drawings among extracted images.

        Args:
        ----
            image_documents: List of image documents with metadata.

        Returns:
        -------
            List of identified technical drawings with enhanced metadata.

        """
        drawings = []

        # Try to identify technical drawings based on image characteristics
        for img_doc in image_documents:
            metadata = img_doc["metadata"]
            image_path = metadata.get("image_path")

            if not image_path:
                continue

            try:
                # Use basic heuristics to identify drawings
                # In a real implementation, you might use a more sophisticated
                # image classification approach
                is_drawing = self._check_if_drawing(image_path)

                if is_drawing:
                    # Create a copy of the document with enhanced metadata
                    drawing_doc = img_doc.copy()
                    drawing_doc["metadata"] = metadata.copy()
                    drawing_doc["metadata"]["filetype"] = "technical_drawing"
                    drawing_doc["metadata"]["content_type"] = "drawing"

                    # Add a more descriptive caption
                    drawing_doc["content"] = f"Technical drawing from page {metadata.get('page_num', 'unknown')}"

                    drawings.append(drawing_doc)
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")

        return drawings

    def _check_if_drawing(self, image_path: str) -> bool:
        """Check if an image is likely a technical drawing.

        This is a simplified heuristic. In a real implementation, you might use
        a machine learning model trained to identify technical drawings.

        Args:
        ----
            image_path: Path to the image file.

        Returns:
        -------
            True if the image is likely a technical drawing, False otherwise.

        """
        try:
            import numpy as np
            from PIL import Image

            # Open the image
            img = Image.open(image_path)

            # Convert to grayscale
            gray_img = img.convert("L")

            # Convert to numpy array
            img_array = np.array(gray_img)

            # Calculate image statistics
            mean_val = np.mean(img_array)
            std_val = np.std(img_array)

            # Heuristics for technical drawings:
            # - High contrast (high standard deviation)
            # - Mostly white background (high mean value)
            # - Limited color palette
            is_high_contrast = std_val > 50
            is_light_background = mean_val > 200

            # Check if the image has limited colors (typical for line drawings)
            colors = len(np.unique(img_array))
            has_limited_colors = colors < 20

            # Combine heuristics
            return is_high_contrast and (is_light_background or has_limited_colors)

        except Exception as e:
            print(f"Error analyzing image {image_path}: {str(e)}")
            return False

    def load(self) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """Load the DIN standard PDF with specialized extraction.

        Returns
        -------
            List of documents including structured sections, tables, and images/drawings.

        """
        documents = []

        # Extract text content
        text_content = self._extract_text(self.file_path)

        # Identify and extract sections
        section_documents = self._identify_sections(text_content)
        documents.extend(section_documents)

        # Extract and add tables if requested
        if self.extract_tables:
            table_documents = self._extract_tables(self.file_path)
            documents.extend(table_documents)

        # Extract images if requested
        image_documents = []
        if self.extract_images:
            image_documents = self._extract_images(self.file_path)
            documents.extend(image_documents)

        # Identify technical drawings if requested
        if self.extract_drawings and image_documents:
            drawing_documents = self._identify_drawings(image_documents)
            documents.extend(drawing_documents)

        return documents


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

    def load(self) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """Load the CSV file.

        Returns
        -------
            List of documents, one per row in the CSV file.

        """
        documents: List[Dict[str, Union[str, Dict[str, Any]]]] = []

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
                metadata: Dict[str, Any] = {
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
        use_enhanced_pdf: bool = False,
        din_standard_mode: bool = False,
    ):
        """Initialize the loader.

        Args:
        ----
            directory_path: Path to the directory.
            recursive: Whether to recursively load files from subdirectories.
            glob_pattern: Pattern to match files (e.g., "*.txt").
            use_enhanced_pdf: Whether to use the enhanced PDF loader for PDF files.
            din_standard_mode: Whether to use the DIN standard loader for PDF files.

        """
        self.directory_path = Path(directory_path)
        if not self.directory_path.exists() or not self.directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        self.recursive = recursive
        self.glob_pattern = glob_pattern
        self.use_enhanced_pdf = use_enhanced_pdf
        self.din_standard_mode = din_standard_mode

    def load(self) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """Load documents from the directory.

        Returns
        -------
            List of documents from all matching files in the directory.

        """
        documents: List[Dict[str, Union[str, Dict[str, Any]]]] = []

        # Determine the glob pattern
        pattern = self.glob_pattern or "*.*"

        # Get all matching files
        if self.recursive:
            # Use ** for recursive matching
            if "/" not in pattern and "\\" not in pattern:
                # If pattern doesn't include path separators, prepend **/ for recursion
                search_pattern = f"**/{pattern}"
            else:
                search_pattern = pattern

            matching_files = list(self.directory_path.glob(search_pattern))
        else:
            matching_files = list(self.directory_path.glob(pattern))

        # Filter out directories
        matching_files = [f for f in matching_files if f.is_file()]

        # Load each file
        for file_path in matching_files:
            try:
                # Select appropriate loader based on file extension
                extension = file_path.suffix.lower()
                loader: DocumentLoader

                if extension == ".csv":
                    loader = CSVLoader(file_path)
                elif extension == ".pdf":
                    if self.din_standard_mode:
                        # Create image output directory based on PDF name
                        image_dir = file_path.parent / f"{file_path.stem}_images"
                        loader = DINStandardLoader(file_path, image_output_dir=image_dir)
                    elif self.use_enhanced_pdf:
                        # Create image output directory based on PDF name
                        image_dir = file_path.parent / f"{file_path.stem}_images"
                        loader = EnhancedPDFLoader(file_path, image_output_dir=image_dir)
                    else:
                        loader = PDFLoader(file_path)
                elif extension in [".txt", ".md", ".html", ".json"]:
                    loader = TextFileLoader(file_path)
                else:
                    print(f"Unsupported file type: {extension}. Skipping {file_path}")
                    continue

                # Load the file and add documents to the list
                file_documents = loader.load()
                documents.extend(file_documents)

            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

        return documents
