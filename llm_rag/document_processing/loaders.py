"""Document loaders for the RAG system."""

import csv
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
            raise ImportError("PyPDF2 is required for PDF loading. " "Install it with 'pip install PyPDF2'") from err

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
    ):
        """Initialize the loader.

        Args:
        ----
            directory_path: Path to the directory.
            recursive: Whether to recursively load files from subdirectories.
            glob_pattern: Pattern to match files (e.g., "*.txt").

        """
        self.directory_path = Path(directory_path)
        if not self.directory_path.exists() or not self.directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        self.recursive = recursive
        self.glob_pattern = glob_pattern

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
                    loader = PDFLoader(file_path)
                elif extension in [".txt", ".md", ".html", ".json"]:
                    loader = TextFileLoader(file_path)
                else:
                    print(f"Unsupported file type: {extension}. " f"Skipping {file_path}")
                    continue

                # Load the file and add documents to the list
                file_documents = loader.load()
                documents.extend(file_documents)

            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

        return documents
