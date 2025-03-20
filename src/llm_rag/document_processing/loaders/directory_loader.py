"""Directory-based document loader.

This module provides components for loading documents from directories,
with support for recursive traversal and file type detection.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from llm_rag.utils.logging import get_logger

from .base import (
    DocumentLoader,
    Documents,
    registry,  # Import the registry
)

logger = get_logger(__name__)


def load_document(file_path: Union[str, Path]) -> Documents:
    """Load a document from a file using the appropriate loader.

    Args:
        file_path: Path to the file

    Returns:
        List of documents loaded from the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If no loader is found for the file type

    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Use the registry to find an appropriate loader
    loader = registry.create_loader_for_file(path)

    # Raise error if no loader is found
    if loader is None:
        raise ValueError(f"No loader found for file type: {path.suffix}")

    # Load the document using the loader
    return loader.load_from_file(path)


def load_documents_from_directory(directory_path: str) -> List[dict]:
    """Load all documents from a directory using appropriate loaders.

    Args:
        directory_path: Path to the directory containing documents

    Returns:
        List of loaded documents

    Raises:
        NotADirectoryError: If the directory does not exist or is not a directory

    """
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory_path}")

    loader = DirectoryLoader(directory_path)
    return loader.load()


class DirectoryLoader(DocumentLoader):
    """Loader for directories of files."""

    def __init__(
        self,
        directory_path: Union[str, Path],
        glob_pattern: str = "*.*",
        recursive: bool = True,
        loader_mapping: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ):
        """Initialize the DirectoryLoader.

        Args:
            directory_path: Path to the directory
            glob_pattern: Pattern to match files (default: "*.*")
            recursive: Whether to search recursively (default: True)
            loader_mapping: Optional mapping of file extensions to loader classes
            **kwargs: Additional arguments passed to child loaders

        """
        self.directory_path = Path(directory_path)
        self.glob_pattern = glob_pattern
        self.recursive = recursive
        self.loader_mapping = loader_mapping or {}
        self.kwargs = kwargs

        if not self.directory_path.exists():
            logger.warning(f"Directory not found: {self.directory_path}")
        elif not self.directory_path.is_dir():
            logger.warning(f"Path is not a directory: {self.directory_path}")

    def load(self) -> Documents:
        """Load documents from files in the directory.

        Returns:
            List of documents from all files in the directory

        Raises:
            NotADirectoryError: If the directory does not exist or is not a directory

        """
        try:
            logger.info(f"Loading documents from directory: {self.directory_path}")

            if not self.directory_path.exists() or not self.directory_path.is_dir():
                logger.error(f"Invalid directory path: {self.directory_path}")
                raise NotADirectoryError(f"Invalid directory path: {self.directory_path}")

            documents = []

            # Handle recursive vs. non-recursive
            if self.recursive:
                glob_path = self.directory_path.glob("**/" + self.glob_pattern)
            else:
                glob_path = self.directory_path.glob(self.glob_pattern)

            # Process each file
            for file_path in glob_path:
                if not file_path.is_file():
                    continue

                try:
                    # First try to use registry to find a loader
                    loader = registry.create_loader_for_file(file_path)

                    # If no loader found in registry, fall back to manual detection
                    if loader is None:
                        # Determine which loader to use based on file extension
                        ext = file_path.suffix.lower()

                        # Check if we have a specific loader for this extension
                        if ext in self.loader_mapping:
                            loader_cls = self.loader_mapping[ext]
                            loader = loader_cls(file_path, **self.kwargs)
                        else:
                            # Try to determine the right loader based on extension
                            loader = self._get_loader_for_extension(ext, file_path)

                    # Load the file - use load_from_file for loaders from registry
                    if hasattr(loader, "load_from_file"):
                        file_docs = loader.load_from_file(file_path)
                    else:
                        file_docs = loader.load()
                    documents.extend(file_docs)

                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")

            return documents
        except NotADirectoryError:
            raise
        except Exception as e:
            logger.error(f"Error loading directory {self.directory_path}: {e}")
            return []

    def _get_loader_for_extension(self, ext: str, file_path: Path) -> DocumentLoader:
        """Get an appropriate loader for the given file extension.

        Args:
            ext: File extension (including dot)
            file_path: Path to the file

        Returns:
            An instance of an appropriate DocumentLoader

        """
        from .file_loaders import CSVLoader, JSONLoader, PDFLoader, TextFileLoader, XMLLoader

        # Match extensions to appropriate loaders
        if ext in [".txt", ".md", ".rst", ".log"]:
            return TextFileLoader(file_path, **self.kwargs)
        elif ext == ".pdf":
            return PDFLoader(file_path, **self.kwargs)
        elif ext == ".csv":
            return CSVLoader(file_path, **self.kwargs)
        elif ext == ".json":
            return JSONLoader(file_path, **self.kwargs)
        elif ext == ".xml":
            return XMLLoader(file_path, **self.kwargs)
        else:
            # Default to text for unknown types
            logger.warning(f"No specific loader for {ext} files. Using TextFileLoader.")
            return TextFileLoader(file_path, **self.kwargs)
