"""File-based document loaders.

This module provides document loaders for various file types including text, CSV, and PDF files.
"""

import csv
import importlib.util
import logging
from pathlib import Path
from typing import List, Optional, Union

from ..processors import Documents
from .base import DocumentLoader, FileLoader, registry

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


# Register the loaders
registry.register(TextFileLoader, extensions=["txt", "text", "md", "markdown"])
registry.register(CSVLoader, extensions=["csv"])
