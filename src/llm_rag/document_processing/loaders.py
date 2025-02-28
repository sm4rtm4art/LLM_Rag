"""Document loader module for the llm-rag package.

This module provides utilities for loading documents from various sources.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class DocumentLoader(ABC):
    """Abstract base class for document loaders.

    Document loaders are responsible for loading documents from various sources
    and converting them into a standard format for further processing.
    """

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """Load documents from the source.

        Returns
        -------
            A list of documents, where each document is a dictionary with
            at least 'content' and 'metadata' keys.

        """
        pass


class TextFileLoader(DocumentLoader):
    """Loader for plain text files."""

    def __init__(self, file_path: Union[str, Path]) -> None:
        """Initialize the text file loader.

        Args:
        ----
            file_path: Path to the text file to load.

        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def load(self) -> List[Dict[str, Any]]:
        """Load the text file as a document.

        Returns
        -------
            A list containing a single document dictionary with 'content'
            and 'metadata' keys.

        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        metadata = {
            "source": str(self.file_path),
            "filename": self.file_path.name,
            "filetype": "text",
        }

        return [{"content": content, "metadata": metadata}]


class CSVLoader(DocumentLoader):
    """Loader for CSV files."""

    def __init__(
        self,
        file_path: Union[str, Path],
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ) -> None:
        """Initialize the CSV loader.

        Args:
        ----
            file_path: Path to the CSV file to load.
            content_columns: List of column names to include in the document
                content. If None, all columns except metadata_columns will be
                used.
            metadata_columns: List of column names to include as metadata.
                If None, no columns will be used as metadata.

        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.content_columns = content_columns
        self.metadata_columns = metadata_columns or []

    def load(self) -> List[Dict[str, Any]]:
        """Load the CSV file as documents.

        Each row in the CSV file becomes a separate document.

        Returns
        -------
            A list of document dictionaries, one per row.

        """
        df = pd.read_csv(self.file_path)

        # Determine content columns if not specified
        if self.content_columns is None:
            self.content_columns = [col for col in df.columns if col not in self.metadata_columns]

        documents = []
        for _, row in df.iterrows():
            # Combine content columns into a single string
            content_parts = []
            for col in self.content_columns:
                if col in row and pd.notna(row[col]):
                    content_parts.append(f"{col}: {row[col]}")

            content = "\n".join(content_parts)

            # Extract metadata from specified columns
            metadata = {
                "source": str(self.file_path),
                "filename": self.file_path.name,
                "filetype": "csv",
            }

            for col in self.metadata_columns:
                if col in row and pd.notna(row[col]):
                    metadata[col] = str(row[col])

            documents.append({"content": content, "metadata": metadata})

        return documents


class DirectoryLoader:
    """Loader for loading all supported files in a directory."""

    def __init__(
        self,
        directory_path: Union[str, Path],
        glob_pattern: str = "*.*",
        recursive: bool = False,
    ) -> None:
        """Initialize the directory loader.

        Args:
        ----
            directory_path: Path to the directory to load files from.
            glob_pattern: Pattern for matching files.
            recursive: Whether to search subdirectories recursively.

        """
        self.directory_path = Path(directory_path)
        if not self.directory_path.exists() or not self.directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        self.glob_pattern = glob_pattern
        self.recursive = recursive

        # Map file extensions to appropriate loaders
        self.extension_map = {
            ".txt": TextFileLoader,
            ".csv": CSVLoader,
            # Add more mappings as needed
        }

    def load(self) -> List[Dict[str, Any]]:
        """Load all supported files in the directory.

        Returns
        -------
            A list of document dictionaries from all loaded files.

        """
        documents = []

        if self.recursive:
            paths = self.directory_path.rglob(self.glob_pattern)
        else:
            paths = self.directory_path.glob(self.glob_pattern)

        for path in paths:
            if path.is_file():
                extension = path.suffix.lower()
                if extension in self.extension_map:
                    loader_class = self.extension_map[extension]
                    try:
                        loader = loader_class(path)
                        docs = loader.load()
                        documents.extend(docs)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")

        return documents
