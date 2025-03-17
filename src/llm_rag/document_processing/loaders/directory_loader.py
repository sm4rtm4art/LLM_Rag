"""Directory document loader.

This module provides a loader for loading documents from a directory.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..processors import Documents
from .base import DocumentLoader, registry

logger = logging.getLogger(__name__)


class DirectoryLoader(DocumentLoader):
    """Load documents from all files in a directory.

    This loader finds all files in a directory that match a specified pattern,
    and loads each file using the appropriate loader based on its extension.
    """

    def __init__(
        self,
        directory_path: Union[str, Path],
        glob_pattern: str = "*.*",
        recursive: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        exclude_hidden: bool = True,
        loader_kwargs: Optional[Dict] = None,
    ):
        """Initialize the directory loader.

        Parameters
        ----------
        directory_path : Union[str, Path]
            Path to the directory to load files from.
        glob_pattern : str, optional
            Pattern to match files, by default "*.*"
        recursive : bool, optional
            Whether to search recursively in subdirectories, by default False
        exclude_patterns : Optional[List[str]], optional
            Patterns to exclude, by default None
        exclude_hidden : bool, optional
            Whether to exclude hidden files, by default True
        loader_kwargs : Optional[Dict], optional
            Additional arguments to pass to the loaders, by default None

        """
        self.directory_path = Path(directory_path)
        self.glob_pattern = glob_pattern
        self.recursive = recursive
        self.exclude_patterns = exclude_patterns or []
        self.exclude_hidden = exclude_hidden
        self.loader_kwargs = loader_kwargs or {}

    def load(self) -> Documents:
        """Load documents from all files in the directory.

        Returns
        -------
        Documents
            List of documents loaded from all files.

        Raises
        ------
        NotADirectoryError
            If the directory path is not a valid directory.

        """
        if not self.directory_path.exists() or not self.directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {self.directory_path}")

        return self.load_from_directory(
            self.directory_path,
            self.glob_pattern,
            self.recursive,
            self.exclude_patterns,
            self.exclude_hidden,
            self.loader_kwargs,
        )

    @staticmethod
    def load_from_directory(
        directory_path: Union[str, Path],
        glob_pattern: str = "*.*",
        recursive: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        exclude_hidden: bool = True,
        loader_kwargs: Optional[Dict] = None,
    ) -> Documents:
        """Load documents from all files in a directory.

        Parameters
        ----------
        directory_path : Union[str, Path]
            Path to the directory to load files from.
        glob_pattern : str, optional
            Pattern to match files, by default "*.*"
        recursive : bool, optional
            Whether to search recursively in subdirectories, by default False
        exclude_patterns : Optional[List[str]], optional
            Patterns to exclude, by default None
        exclude_hidden : bool, optional
            Whether to exclude hidden files, by default True
        loader_kwargs : Optional[Dict], optional
            Additional arguments to pass to the loaders, by default None

        Returns
        -------
        Documents
            List of documents loaded from all files.

        Raises
        ------
        NotADirectoryError
            If the directory path is not a valid directory.

        """
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        exclude_patterns = exclude_patterns or []
        loader_kwargs = loader_kwargs or {}

        # Get all files matching the glob pattern
        if recursive:
            files = list(directory_path.rglob(glob_pattern))
        else:
            files = list(directory_path.glob(glob_pattern))

        # Filter out directories and hidden files
        files = [f for f in files if f.is_file() and (not exclude_hidden or not f.name.startswith("."))]

        # Apply exclude patterns
        for pattern in exclude_patterns:
            files = [f for f in files if not f.match(pattern)]

        # Load documents from each file
        documents = []
        for file_path in files:
            try:
                loader = registry.create_loader_for_file(file_path)
                if loader is not None:
                    file_docs = loader.load_from_file(file_path, **loader_kwargs)
                    documents.extend(file_docs)
                else:
                    logger.warning(f"No loader found for file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
                continue

        return documents


# Register the loader
registry.register(DirectoryLoader)
