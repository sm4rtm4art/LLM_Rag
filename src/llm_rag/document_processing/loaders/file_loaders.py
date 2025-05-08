"""File-based document loaders for the LLM-RAG system."""

import csv
import importlib.util
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    logger.warning('Pandas not available. Some CSV/Excel loading capabilities will be affected.')

# Check for PyMuPDF availability
PYMUPDF_AVAILABLE = importlib.util.find_spec('fitz') is not None
if not PYMUPDF_AVAILABLE:
    logger.warning('PyMuPDF not available. PDF loading capabilities will be limited.')

# Check for pypdf availability (formerly PyPDF2)
PYPDF_AVAILABLE = importlib.util.find_spec('pypdf') is not None
if not PYPDF_AVAILABLE:
    logger.warning('pypdf not available. PDF loading capabilities will be limited.')

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
            'Using xml.etree.ElementTree for XML parsing, which is vulnerable to XML attacks. '
            'Install defusedxml package for secure XML parsing.',
            SecurityWarning,
            stacklevel=2,
        )
        HAS_SECURE_XML = False
    HAS_XML = True
    # For backward compatibility with tests
    XML_AVAILABLE = HAS_XML
except ImportError:
    HAS_XML = False
    # For backward compatibility with tests
    XML_AVAILABLE = HAS_XML
    logger.warning('XML processing libraries not available. XML loading capabilities will be affected.')


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
            raise ValueError('No file path provided. Either initialize with a file path or use load_from_file.')

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
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Create metadata with file information
            metadata = {
                'source': str(file_path),
                'filename': file_path.name,
                'filetype': 'text',
                'creation_date': None,  # Could add file stats here
            }

            # Return a list with a single document
            return [{'content': content, 'metadata': metadata}]
        except Exception as e:
            logger.error(f'Error loading text file {file_path}: {e}')
            raise


class CSVLoader(DocumentLoader, FileLoader):
    """Load documents from CSV files."""

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        delimiter: str = ',',
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
            raise ValueError('No file path provided. Either initialize with a file path or use load_from_file.')

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
            logger.error(f'Error loading CSV file {file_path}: {e}')
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
                content = ' '.join(str(row[col]) for col in self.content_columns if col in row)
            else:
                # Use all columns as content
                content = ' '.join(f'{col}: {val}' for col, val in row.items())

            # Create metadata
            metadata = {
                'source': str(file_path),
                'filename': file_path.name,
                'filetype': 'csv',
                'row_index': _,
            }

            # Add specific metadata columns if requested
            if self.metadata_columns:
                for col in self.metadata_columns:
                    if col in row:
                        metadata[col] = row[col]

            documents.append({'content': content, 'metadata': metadata})

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

        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=self.delimiter)

            for i, row in enumerate(reader):
                # Handle content columns
                if self.content_columns:
                    # Create content from specified columns
                    content = ' '.join(str(row.get(col, '')) for col in self.content_columns)
                else:
                    # Use all columns as content
                    content = ' '.join(f'{col}: {val}' for col, val in row.items())

                # Create metadata
                metadata = {
                    'source': str(file_path),
                    'filename': file_path.name,
                    'filetype': 'csv',
                    'row_index': i,
                }

                # Add specific metadata columns if requested
                if self.metadata_columns:
                    for col in self.metadata_columns:
                        if col in row:
                            metadata[col] = row[col]

                documents.append({'content': content, 'metadata': metadata})

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
            raise ValueError('No file path provided. Either initialize with a file path or use load_from_file.')

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

        Raises
        ------
        ImportError
            If XML processing libraries are not available.
        FileNotFoundError
            If the file does not exist.

        """
        if not XML_AVAILABLE:
            raise ImportError('XML processing libraries not available. Install required dependencies.')

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')

        try:
            if not HAS_SECURE_XML:
                warnings.warn(
                    'Using xml.etree.ElementTree for XML parsing, which is vulnerable to XML attacks. '
                    'Install defusedxml package for secure XML parsing.',
                    SecurityWarning,
                    stacklevel=2,
                )
            tree = ET.parse(file_path)  # nosec B314
            root = tree.getroot()

            # Generate metadata common to all documents from this file
            base_metadata = {
                'source': str(file_path),
                'filename': file_path.name,
                'filetype': 'xml',
            }

            # If splitting by tag, create multiple documents
            if self.split_by_tag:
                return self._process_split_documents(root, base_metadata)
            else:
                # Create a single document with all content
                return self._process_whole_document(root, base_metadata)

        except Exception as e:
            logger.error(f'Error parsing XML file {file_path}: {e}')
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
                elements = root.findall(f'.//{tag}')
                if elements:
                    # Just take the first occurrence as metadata
                    metadata[tag] = elements[0].text.strip() if elements[0].text else ''

                    # Include attributes if specified
                    if self.flatten_attributes and elements[0].attrib:
                        for attr_name, attr_value in elements[0].attrib.items():
                            metadata[f'{tag}_{attr_name}'] = attr_value

        return [{'content': content, 'metadata': metadata}]

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
        elements = root.findall(f'.//{self.split_by_tag}')

        for i, element in enumerate(elements):
            # Extract content from this element
            content = self._extract_content(element)

            # Create metadata for this document
            metadata = base_metadata.copy()
            metadata['index'] = i

            # Add any specific metadata from tags within this element
            if self.metadata_tags:
                for tag in self.metadata_tags:
                    sub_elements = element.findall(f'.//{tag}')
                    if sub_elements:
                        metadata[tag] = sub_elements[0].text.strip() if sub_elements[0].text else ''

                        # Include attributes if specified
                        if self.flatten_attributes and sub_elements[0].attrib:
                            for attr_name, attr_value in sub_elements[0].attrib.items():
                                metadata[f'{tag}_{attr_name}'] = attr_value

            documents.append({'content': content, 'metadata': metadata})

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
            elements = element.findall(f'.//{self.text_tag}')
            text_parts = []

            for el in elements:
                if self.include_tags_in_text:
                    text_parts.append(ET.tostring(el, encoding='unicode'))
                else:
                    if el.text:
                        text_parts.append(el.text.strip())

            return '\n\n'.join(text_parts)

        elif self.content_tags:
            # Extract text from the specified content tags
            text_parts = []

            for tag in self.content_tags:
                elements = element.findall(f'.//{tag}')
                for el in elements:
                    if self.include_tags_in_text:
                        text_parts.append(ET.tostring(el, encoding='unicode'))
                    else:
                        if el.text:
                            # Add the tag name as a prefix
                            tag_text = el.text.strip()
                            if tag_text:
                                text_parts.append(f'{tag}: {tag_text}')

            return '\n\n'.join(text_parts)

        else:
            # Extract all text content
            if self.include_tags_in_text:
                return ET.tostring(element, encoding='unicode')
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
                return '\n'.join(filter(None, text_parts))


class PDFLoader(DocumentLoader, FileLoader):
    """Load documents from PDF files."""

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        extract_images: bool = False,
        password: Optional[str] = None,
        extract_tables: bool = False,
    ):
        """Initialize the PDF loader.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], optional
            Path to the PDF file, by default None
        extract_images : bool, optional
            Whether to extract images from the PDF, by default False
        password : Optional[str], optional
            Password to decrypt the PDF, by default None
        extract_tables : bool, optional
            Whether to extract tables from the PDF, by default False

        """
        self.file_path = Path(file_path) if file_path else None
        self.extract_images = extract_images
        self.password = password
        self.extract_tables = extract_tables

    def load(self) -> Documents:
        """Load documents from the PDF file specified during initialization.

        Returns
        -------
        Documents
            List of documents extracted from the PDF.

        Raises
        ------
        ValueError
            If file_path was not provided during initialization.

        """
        if not self.file_path:
            raise ValueError('No file path provided. Either initialize with a file path or use load_from_file.')

        return self.load_from_file(self.file_path)

    def load_from_file(self, file_path: Union[str, Path]) -> Documents:
        """Load documents from a PDF file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the PDF file.

        Returns
        -------
        Documents
            List of documents, typically one per page.

        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')

        try:
            if PYPDF_AVAILABLE:
                return self._load_with_pypdf(file_path)
            elif PYMUPDF_AVAILABLE:
                return self._load_with_pymupdf(file_path)
            else:
                raise ImportError('No PDF processing libraries available. Install pypdf or PyMuPDF.')
        except Exception as e:
            logger.error(f'Error loading PDF file {file_path}: {e}')
            raise

    def _load_with_pypdf(self, file_path: Path) -> Documents:
        """Load PDF using pypdf.

        Parameters
        ----------
        file_path : Path
            Path to the PDF file.

        Returns
        -------
        Documents
            List of documents, one per page.

        """
        import pypdf

        with open(file_path, 'rb') as f:
            pdf = pypdf.PdfReader(f)
            if self.password:
                pdf.decrypt(self.password)

            documents = []
            metadata = {
                'source': str(file_path),
                'filename': file_path.name,
                'filetype': 'pdf',
                'total_pages': len(pdf.pages),
            }

            # Add PDF metadata if available
            if pdf.metadata:
                for key, value in pdf.metadata.items():
                    if key.startswith('/'):
                        key = key[1:]  # Remove leading slash from keys
                    metadata[f'pdf_{key.lower()}'] = str(value)

            # Process each page
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    text = f'[No extractable text on page {i + 1}]'

                page_metadata = metadata.copy()
                page_metadata['page_number'] = i + 1

                documents.append({'content': text, 'metadata': page_metadata})

            return documents

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
        import fitz  # PyMuPDF

        documents = []
        pdf = fitz.open(file_path)

        if self.password:
            if pdf.authenticate(self.password) != 0:
                raise ValueError('Invalid PDF password')

        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'filetype': 'pdf',
            'total_pages': len(pdf),
        }

        # Add PDF metadata if available
        pdf_metadata = pdf.metadata
        if pdf_metadata:
            for key, value in pdf_metadata.items():
                if value:
                    metadata[f'pdf_{key.lower()}'] = str(value)

        # Process each page
        for i in range(len(pdf)):
            page = pdf[i]
            text = page.get_text()
            if not text:
                text = f'[No extractable text on page {i + 1}]'

            page_metadata = metadata.copy()
            page_metadata['page_number'] = i + 1

            documents.append({'content': text, 'metadata': page_metadata})

        return documents


class EnhancedPDFLoader(PDFLoader):
    """Enhanced PDF loader with additional extraction capabilities."""

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        extract_images: bool = True,
        extract_tables: bool = True,
        password: Optional[str] = None,
        combine_pages: bool = False,
        use_ocr: bool = False,
        ocr_languages: str = 'eng',
    ):
        """Initialize the enhanced PDF loader.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], optional
            Path to the PDF file, by default None
        extract_images : bool, optional
            Whether to extract images from the PDF, by default True
        extract_tables : bool, optional
            Whether to extract tables from the PDF, by default True
        password : Optional[str], optional
            Password to decrypt the PDF, by default None
        combine_pages : bool, optional
            Whether to combine all pages into a single document, by default False
        use_ocr : bool, optional
            Whether to use OCR on images for text extraction, by default False
        ocr_languages : str, optional
            Language code(s) for OCR, by default "eng"

        """
        super().__init__(file_path, extract_images, password, extract_tables)
        self.combine_pages = combine_pages
        self.use_ocr = use_ocr
        self.ocr_languages = ocr_languages

    def load_from_file(self, file_path: Union[str, Path]) -> Documents:
        """Load documents from a PDF file with enhanced processing.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the PDF file.

        Returns
        -------
        Documents
            List of documents from the PDF.

        """
        # First, get the basic documents using the parent class
        documents = super().load_from_file(file_path)

        # If combining pages, merge all content into a single document
        if self.combine_pages:
            combined_content = '\n\n'.join(doc['content'] for doc in documents)
            combined_metadata = documents[0]['metadata'].copy()
            combined_metadata.pop('page_number', None)
            return [{'content': combined_content, 'metadata': combined_metadata}]

        # If extracting tables and PyMuPDF is available, add table information
        if self.extract_tables and PYMUPDF_AVAILABLE:
            try:
                import fitz

                file_path = Path(file_path)
                pdf = fitz.open(file_path)

                if self.password:
                    if pdf.authenticate(self.password) != 0:
                        raise ValueError('Invalid PDF password')

                for i, doc in enumerate(documents):
                    if i < len(pdf):
                        page = pdf[i]
                        # Extract tables (basic implementation)
                        tables = page.find_tables()
                        if tables and tables.tables:
                            table_texts = []
                            for table in tables:
                                table_text = []
                                for row in table.rows:
                                    row_text = []
                                    for cell in row:
                                        if cell.text:
                                            row_text.append(cell.text)
                                    if row_text:
                                        table_text.append(' | '.join(row_text))
                                if table_text:
                                    table_texts.append('\n'.join(table_text))

                            if table_texts:
                                doc['content'] += '\n\nTables:\n' + '\n\n'.join(table_texts)
                                doc['metadata']['has_tables'] = True
                                doc['metadata']['table_count'] = len(table_texts)

            except Exception as e:
                logger.warning(f'Error extracting tables: {e}')

        return documents


class JSONLoader(DocumentLoader, FileLoader):
    """Load documents from JSON files."""

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        jq_schema: str = '.',
        content_key: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        text_content: bool = True,
    ):
        """Initialize the JSON loader.

        Parameters
        ----------
        file_path : Optional[Union[str, Path]], optional
            Path to the JSON file, by default None
        jq_schema : str, optional
            JQ-like schema for extracting content, by default "." (entire document)
        content_key : Optional[str], optional
            Key to use for document content, by default None
        metadata_keys : Optional[List[str]], optional
            Keys to include as metadata, by default None
        text_content : bool, optional
            Whether to convert content to text, by default True

        """
        self.file_path = Path(file_path) if file_path else None
        self.jq_schema = jq_schema
        self.content_key = content_key
        self.metadata_keys = metadata_keys
        self.text_content = text_content

    def load(self) -> Documents:
        """Load documents from the JSON file specified during initialization.

        Returns
        -------
        Documents
            List of documents extracted from the JSON file.

        Raises
        ------
        ValueError
            If file_path was not provided during initialization.

        """
        if not self.file_path:
            raise ValueError('No file path provided. Either initialize with a file path or use load_from_file.')

        return self.load_from_file(self.file_path)

    def load_from_file(self, file_path: Union[str, Path]) -> Documents:
        """Load documents from a JSON file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the JSON file.

        Returns
        -------
        Documents
            List of documents extracted from the JSON file.

        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')

        try:
            import json

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Apply jq-like schema (basic implementation)
            if self.jq_schema != '.':
                # Split by dots and apply each key
                keys = self.jq_schema.strip('.').split('.')
                current_data = data
                for key in keys:
                    if key in current_data:
                        current_data = current_data[key]
                    else:
                        raise ValueError(f"Key '{key}' not found in JSON structure")
                data = current_data

            documents = []

            # Handle different JSON structures
            if isinstance(data, list):
                # Process each item in the list
                for i, item in enumerate(data):
                    doc = self._process_item(item, i, file_path)
                    if doc:
                        documents.append(doc)
            else:
                # Process the single item
                doc = self._process_item(data, 0, file_path)
                if doc:
                    documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f'Error loading JSON file {file_path}: {e}')
            raise

    def _process_item(self, item: Any, index: int, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single JSON item into a document.

        Parameters
        ----------
        item : Any
            The JSON item to process.
        index : int
            Index of the item (for lists).
        file_path : Path
            Path to the source file.

        Returns
        -------
        Optional[Dict[str, Any]]
            Document dictionary or None if item couldn't be processed.

        """
        # Extract content based on content_key if specified
        if self.content_key and isinstance(item, dict):
            if self.content_key in item:
                content = item[self.content_key]
            else:
                logger.warning(f"Content key '{self.content_key}' not found in item {index}")
                return None
        else:
            content = item

        # Convert content to string if needed
        if self.text_content and not isinstance(content, str):
            import json

            content = json.dumps(content)

        # Prepare metadata
        metadata = {
            'source': str(file_path),
            'filename': file_path.name,
            'filetype': 'json',
            'index': index,
        }

        # Add specific metadata if requested
        if self.metadata_keys and isinstance(item, dict):
            for key in self.metadata_keys:
                if key in item:
                    metadata[key] = item[key]

        return {'content': content, 'metadata': metadata}


# Register the loaders
registry.register(TextFileLoader, extensions=['txt', 'text', 'md', 'rst', 'log'])
registry.register(CSVLoader, extensions=['csv', 'tsv'])
registry.register(XMLLoader, extensions=['xml'])
registry.register(PDFLoader, extensions=['pdf'])
registry.register(EnhancedPDFLoader, extensions=['pdf'])
registry.register(JSONLoader, extensions=['json'])
