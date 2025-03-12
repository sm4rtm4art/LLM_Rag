"""Factory functions for document loaders.

This module provides factory functions and utilities for creating and using document loaders.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..processors import Documents
from .base import registry

logger = logging.getLogger(__name__)


def load_document(file_path: Union[str, Path], **kwargs) -> Optional[Documents]:
    """Load documents from a file using an appropriate loader.

    This function automatically selects a loader based on the file extension.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the file to load.
    **kwargs
        Additional arguments to pass to the loader.

    Returns
    -------
    Optional[Documents]
        List of documents loaded from the file, or None if no loader was found.

    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None

    loader = registry.create_loader_for_file(file_path, **kwargs)

    if loader is None:
        logger.warning(f"No loader found for file type: {file_path.suffix}")
        return None

    try:
        # Check if the loader supports load_from_file
        if hasattr(loader, "load_from_file"):
            return loader.load_from_file(file_path)

        # Fall back to regular load method
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None


def load_documents_from_directory(
    directory_path: Union[str, Path],
    glob_pattern: str = "*.*",
    recursive: bool = True,
    exclude_patterns: Optional[List[str]] = None,
    **kwargs,
) -> Documents:
    """Load documents from all files in a directory.

    Parameters
    ----------
    directory_path : Union[str, Path]
        Path to the directory.
    glob_pattern : str, optional
        Pattern to match files, by default "*.*"
    recursive : bool, optional
        Whether to search recursively in subdirectories, by default True
    exclude_patterns : Optional[List[str]], optional
        Patterns to exclude, by default None
    **kwargs
        Additional arguments to pass to the loaders.

    Returns
    -------
    Documents
        List of documents loaded from all files.

    """
    directory_path = Path(directory_path)

    if not directory_path.exists() or not directory_path.is_dir():
        logger.error(f"Directory not found: {directory_path}")
        return []

    # Get all files matching the pattern
    if recursive:
        files = list(directory_path.glob(f"**/{glob_pattern}"))
    else:
        files = list(directory_path.glob(glob_pattern))

    # Apply exclusion patterns if any
    if exclude_patterns:
        for pattern in exclude_patterns:
            exclude_files = set()
            if recursive:
                exclude_files.update(directory_path.glob(f"**/{pattern}"))
            else:
                exclude_files.update(directory_path.glob(pattern))

            files = [f for f in files if f not in exclude_files]

    # Load documents from each file
    all_documents = []
    for file_path in files:
        try:
            documents = load_document(file_path, **kwargs)
            if documents:
                all_documents.extend(documents)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            # Continue with other files

    return all_documents


def get_available_loader_extensions() -> Dict[str, str]:
    """Get a mapping of available file extensions to loader names.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping file extensions to loader names.

    """
    return {ext: name for ext, name in registry._extension_mapping.items()}
