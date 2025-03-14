"""File-based document loaders for the LLM-RAG system."""

import csv
import importlib.util
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Union

from ..processors import Documents
from .base import DocumentLoader, FileLoader, registry


# Define a security warning class
class SecurityWarning(Warning):
    """Warning for security-related issues."""

    pass


# Get configured logger
logger = logging.getLogger(__name__)

# Optional imports for pandas
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available. Some CSV/Excel loading capabilities will be affected.")

# Check for PyMuPDF availability
PYMUPDF_AVAILABLE = importlib.util.find_spec("fitz") is not None
if not PYMUPDF_AVAILABLE:
    logger.warning("PyMuPDF not available. PDF loading capabilities will be limited.")

# Check for PyPDF2 availability
PYPDF2_AVAILABLE = importlib.util.find_spec("PyPDF2") is not None
if not PYPDF2_AVAILABLE:
    logger.warning("PyPDF2 not available. PDF loading capabilities will be limited.")

# Optional imports for XML processing
try:
    try:
        # Try to use the secure defusedxml package first
        import defusedxml.ElementTree as ET

        HAS_SECURE_XML = True
    except ImportError:
        # Fall back to standard library with a warning
        import warnings
        import xml.etree.ElementTree as ET  # nosec B405

        warnings.warn(
            "Using xml.etree.ElementTree for XML parsing, which is vulnerable to XML attacks. "
            "Install defusedxml package for secure XML parsing.",
            SecurityWarning,
            stacklevel=2,
        )
        HAS_SECURE_XML = False
    HAS_XML = True
except ImportError:
    HAS_XML = False
    logger.warning("XML processing libraries not available. XML loading capabilities will be affected.")


class TextFileLoader(DocumentLoader, FileLoader):
    """Load documents from text files."""

    def __init__(self, file_path: Optional[Union[str, Path]] = None):
        """Initialize the text file loader.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], optional
            Path to the text file, by default None. If None, file_path must be provided to load_from_file.

        """
        self.file_path = Path(file_path) if file_path else None

    def load(self) -> Documents:
        """Load documents from the file specified during initialization.

        Returns
        -------
        Documents
            List containing a single document with the file's contents.

        Raises
        ------
        ValueError
            If file_path was not provided during initialization.

        """
        if not self.file_path:
            raise ValueError("No file path provided. Either initialize with a file path or use load_from_file.")

        return self.load_from_file(self.file_path)

    def load_from_file(self, file_path: Union[str, Path]) -> Documents:
        """Load document from a text file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the text file to load.

        Returns
        -------
        Documents
            List containing a single document with the file's contents.

        """
        file_path = Path(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Create metadata with file information
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "filetype": "text",
                "creation_date": None,  # Could add file stats here
            }

            # Return a list with a single document
            return [{"content": content, "metadata": metadata}]
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise


class CSVLoader(DocumentLoader, FileLoader):
    """Load documents from CSV files."""

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        delimiter: str = ",",
        use_pandas: bool = True,
    ):
        """Initialize the CSV loader.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], optional
            Path to the CSV file, by default None
        content_columns : Optional[List[str]], optional
            Column names to use as content, by default None (uses all columns)
        metadata_columns : Optional[List[str]], optional
            Column names to include as metadata, by default None
        delimiter : str, optional
            CSV delimiter, by default ","
        use_pandas : bool, optional
            Whether to use pandas for loading (if available), by default True

        """
        self.file_path = Path(file_path) if file_path else None
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.delimiter = delimiter
        self.use_pandas = use_pandas and PANDAS_AVAILABLE

    def load(self) -> Documents:
        """Load documents from the CSV file specified during initialization.

        Returns
        -------
        Documents
            List of documents, one per row in the CSV file.

        Raises
        ------
        ValueError
            If file_path was not provided during initialization.

        """
        if not self.file_path:
            raise ValueError("No file path provided. Either initialize with a file path or use load_from_file.")

        return self.load_from_file(self.file_path)

    def load_from_file(self, file_path: Union[str, Path]) -> Documents:
        """Load documents from a CSV file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the CSV file to load.

        Returns
        -------
        Documents
            List of documents, one per row in the CSV file.

        """
        file_path = Path(file_path)

        try:
            if self.use_pandas and PANDAS_AVAILABLE:
                return self._load_with_pandas(file_path)
            else:
                return self._load_with_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise

    def _load_with_pandas(self, file_path: Path) -> Documents:
        """Load CSV using pandas.

        Parameters
        ----------
        file_path : Path
            Path to the CSV file.

        Returns
        -------
        Documents
            List of documents, one per row.

        """
        df = pd.read_csv(file_path, delimiter=self.delimiter)
        documents = []

        for _, row in df.iterrows():
            # Handle content columns
            if self.content_columns:
                # Create content from specified columns
                content = " ".join(str(row[col]) for col in self.content_columns if col in row)
            else:
                # Use all columns as content
                content = " ".join(f"{col}: {val}" for col, val in row.items())

            # Create metadata
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "filetype": "csv",
                "row_index": _,
            }

            # Add specific metadata columns if requested
            if self.metadata_columns:
                for col in self.metadata_columns:
                    if col in row:
                        metadata[col] = row[col]

            documents.append({"content": content, "metadata": metadata})

        return documents

    def _load_with_csv(self, file_path: Path) -> Documents:
        """Load CSV using the csv module.

        Parameters
        ----------
        file_path : Path
            Path to the CSV file.

        Returns
        -------
        Documents
            List of documents, one per row.

        """
        documents = []

        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=self.delimiter)

            for i, row in enumerate(reader):
                # Handle content columns
                if self.content_columns:
                    # Create content from specified columns
                    content = " ".join(str(row.get(col, "")) for col in self.content_columns)
                else:
                    # Use all columns as content
                    content = " ".join(f"{col}: {val}" for col, val in row.items())

                # Create metadata
                metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "filetype": "csv",
                    "row_index": i,
                }

                # Add specific metadata columns if requested
                if self.metadata_columns:
                    for col in self.metadata_columns:
                        if col in row:
                            metadata[col] = row[col]

                documents.append({"content": content, "metadata": metadata})

        return documents


class XMLLoader(DocumentLoader, FileLoader):
    """Load documents from XML files.

    This loader extracts text content from XML files and can structure the content
    based on specified tags or paths.
    """

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        content_tags: Optional[List[str]] = None,
        metadata_tags: Optional[List[str]] = None,
        text_tag: Optional[str] = None,
        flatten_attributes: bool = True,
        split_by_tag: Optional[str] = None,
        include_tags_in_text: bool = False,
    ):
        """Initialize the XML loader.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], optional
            Path to the XML file, by default None
        content_tags : Optional[List[str]], optional
            List of tag names to use as content, by default None (uses all text)
        metadata_tags : Optional[List[str]], optional
            List of tag names to include as metadata, by default None
        text_tag : Optional[str], optional
            The tag that contains the main text content, by default None
        flatten_attributes : bool, optional
            Whether to include tag attributes in the content, by default True
        split_by_tag : Optional[str], optional
            If specified, creates a separate document for each occurrence of this tag,
            by default None
        include_tags_in_text : bool, optional
            Whether to include XML tags in the extracted text, by default False

        """
        self.file_path = Path(file_path) if file_path else None
        self.content_tags = content_tags
        self.metadata_tags = metadata_tags
        self.text_tag = text_tag
        self.flatten_attributes = flatten_attributes
        self.split_by_tag = split_by_tag
        self.include_tags_in_text = include_tags_in_text

    def load(self) -> Documents:
        """Load documents from the XML file specified during initialization.

        Returns
        -------
        Documents
            List of documents extracted from the XML file.

        Raises
        ------
        ValueError
            If file_path was not provided during initialization.

        """
        if not self.file_path:
            raise ValueError("No file path provided. Either initialize with a file path or use load_from_file.")

        return self.load_from_file(self.file_path)

    def load_from_file(self, file_path: Union[str, Path]) -> Documents:
        """Load documents from an XML file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the XML file.

        Returns
        -------
        Documents
            List of documents loaded from the XML file.

        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if not HAS_SECURE_XML:
                warnings.warn(
                    "Using xml.etree.ElementTree for XML parsing, which is vulnerable to XML attacks. "
                    "Install defusedxml package for secure XML parsing.",
                    SecurityWarning,
                    stacklevel=2,
                )
            tree = ET.parse(file_path)  # nosec B314
            root = tree.getroot()

            # Generate metadata common to all documents from this file
            base_metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "filetype": "xml",
            }

            # If splitting by tag, create multiple documents
            if self.split_by_tag:
                return self._process_split_documents(root, base_metadata)
            else:
                # Create a single document with all content
                return self._process_whole_document(root, base_metadata)

        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {e}")
            raise

    def _process_whole_document(self, root, base_metadata: dict) -> Documents:
        """Process the XML as a single document.

        Parameters
        ----------
        root : Element
            The root element of the XML document.
        base_metadata : dict
            Base metadata to include in the document.

        Returns
        -------
        Documents
            A list containing a single document with the XML content.

        """
        content = self._extract_content(root)

        # Add any metadata from specified tags
        metadata = base_metadata.copy()
        if self.metadata_tags:
            for tag in self.metadata_tags:
                elements = root.findall(f".//{tag}")
                if elements:
                    # Just take the first occurrence as metadata
                    metadata[tag] = elements[0].text.strip() if elements[0].text else ""

                    # Include attributes if specified
                    if self.flatten_attributes and elements[0].attrib:
                        for attr_name, attr_value in elements[0].attrib.items():
                            metadata[f"{tag}_{attr_name}"] = attr_value

        return [{"content": content, "metadata": metadata}]

    def _process_split_documents(self, root, base_metadata: dict) -> Documents:
        """Process the XML by splitting it into multiple documents.

        Parameters
        ----------
        root : Element
            The root element of the XML document.
        base_metadata : dict
            Base metadata to include in all documents.

        Returns
        -------
        Documents
            A list of documents, one for each occurrence of the split tag.

        """
        documents = []

        # Find all elements with the specified tag
        elements = root.findall(f".//{self.split_by_tag}")

        for i, element in enumerate(elements):
            # Extract content from this element
            content = self._extract_content(element)

            # Create metadata for this document
            metadata = base_metadata.copy()
            metadata["index"] = i

            # Add any specific metadata from tags within this element
            if self.metadata_tags:
                for tag in self.metadata_tags:
                    sub_elements = element.findall(f".//{tag}")
                    if sub_elements:
                        metadata[tag] = sub_elements[0].text.strip() if sub_elements[0].text else ""

                        # Include attributes if specified
                        if self.flatten_attributes and sub_elements[0].attrib:
                            for attr_name, attr_value in sub_elements[0].attrib.items():
                                metadata[f"{tag}_{attr_name}"] = attr_value

            documents.append({"content": content, "metadata": metadata})

        return documents

    def _extract_content(self, element) -> str:
        """Extract content from an XML element.

        Parameters
        ----------
        element : Element
            The XML element to extract content from.

        Returns
        -------
        str
            The extracted text content.

        """
        if self.text_tag:
            # Extract text from the specified tag
            elements = element.findall(f".//{self.text_tag}")
            text_parts = []

            for el in elements:
                if self.include_tags_in_text:
                    text_parts.append(ET.tostring(el, encoding="unicode"))
                else:
                    if el.text:
                        text_parts.append(el.text.strip())

            return "\n\n".join(text_parts)

        elif self.content_tags:
            # Extract text from the specified content tags
            text_parts = []

            for tag in self.content_tags:
                elements = element.findall(f".//{tag}")
                for el in elements:
                    if self.include_tags_in_text:
                        text_parts.append(ET.tostring(el, encoding="unicode"))
                    else:
                        if el.text:
                            # Add the tag name as a prefix
                            tag_text = el.text.strip()
                            if tag_text:
                                text_parts.append(f"{tag}: {tag_text}")

            return "\n\n".join(text_parts)

        else:
            # Extract all text content
            if self.include_tags_in_text:
                return ET.tostring(element, encoding="unicode")
            else:
                # Get all text without tags
                def extract_text(elem):
                    result = []
                    if elem.text:
                        result.append(elem.text.strip())
                    for child in elem:
                        result.extend(extract_text(child))
                    if elem.tail:
                        result.append(elem.tail.strip())
                    return result

                text_parts = extract_text(element)
                return "\n".join(filter(None, text_parts))


# Register the loaders
registry.register(TextFileLoader, extensions=["txt", "text", "md", "rst", "log"])
registry.register(CSVLoader, extensions=["csv", "tsv"])
registry.register(XMLLoader, extensions=["xml"])
