"""Base classes for document loaders.

This module provides the base DocumentLoader class that all loaders should inherit from,
along with common utilities and type definitions for document loading.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TypeAlias, TypeVar, Union, runtime_checkable

from ..processors import Documents

logger = logging.getLogger(__name__)

# Type aliases for readability
DocumentMetadata: TypeAlias = Dict[str, Any]
DocumentContent: TypeAlias = str
Document: TypeAlias = Dict[str, Union[DocumentContent, DocumentMetadata]]
# Documents TypeAlias is already imported from processors module


class DocumentLoader(ABC):
    """Abstract base class for document loaders.

    All document loaders should inherit from this class and implement the load method.
    """

    @abstractmethod
    def load(self) -> Documents:
        """Load documents from a source.

        Returns
        -------
            List of documents, where each document is a dictionary with
            'content' and 'metadata' keys.

        """
        pass


@runtime_checkable
class FileLoader(Protocol):
    """Protocol for document loaders that load from files."""

    def load_from_file(self, file_path: Union[str, Path]) -> Documents:
        """Load documents from a file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the file to load.

        Returns
        -------
        Documents
            List of documents loaded from the file.

        """
        ...


@runtime_checkable
class DirectoryLoader(Protocol):
    """Protocol for document loaders that load from directories."""

    def load_from_directory(
        self, directory_path: Union[str, Path], glob_pattern: Optional[str] = None, recursive: bool = True
    ) -> Documents:
        """Load documents from a directory.

        Parameters
        ----------
        directory_path : Union[str, Path]
            Path to the directory to load.
        glob_pattern : Optional[str], optional
            Pattern to match files, by default None which matches all files.
        recursive : bool, optional
            Whether to search recursively in subdirectories, by default True.

        Returns
        -------
        Documents
            List of documents loaded from the directory.

        """
        ...


@runtime_checkable
class WebLoader(Protocol):
    """Protocol for document loaders that load from web resources."""

    def load_from_url(self, url: str, headers: Optional[Dict[str, str]] = None) -> Documents:
        """Load documents from a URL.

        Parameters
        ----------
        url : str
            URL to load.
        headers : Optional[Dict[str, str]], optional
            HTTP headers to include in the request, by default None.

        Returns
        -------
        Documents
            List of documents loaded from the URL.

        """
        ...


# Generic type for loader factories
T = TypeVar("T", bound=DocumentLoader)


class LoaderRegistry:
    """Registry for document loaders.

    This class maintains a registry of document loader classes and their associated file extensions.
    It provides factory methods for creating loaders based on file types.
    """

    def __init__(self):
        """Initialize the loader registry."""
        self._loaders = {}
        self._extension_mapping = {}

    def register(self, loader_cls, name: str = None, extensions: List[str] = None):
        """Register a loader class.

        Parameters
        ----------
        loader_cls : type
            Loader class to register.
        name : str, optional
            Name to register the loader under, by default the class name.
        extensions : List[str], optional
            File extensions that this loader can handle, by default None.

        """
        name = name or loader_cls.__name__
        self._loaders[name] = loader_cls

        if extensions:
            for ext in extensions:
                # Handle extensions with or without the leading dot
                ext = ext if ext.startswith(".") else f".{ext}"
                self._extension_mapping[ext.lower()] = name

    def get_loader_class(self, name: str) -> type:
        """Get a loader class by name.

        Parameters
        ----------
        name : str
            Name of the loader.

        Returns
        -------
        type
            Loader class.

        Raises
        ------
        KeyError
            If the loader is not registered.

        """
        if name not in self._loaders:
            raise KeyError(f"Loader {name} not registered")
        return self._loaders[name]

    def create_loader(self, name: str, **kwargs) -> DocumentLoader:
        """Create a loader instance by name.

        Parameters
        ----------
        name : str
            Name of the loader.
        **kwargs
            Additional arguments to pass to the loader constructor.

        Returns
        -------
        DocumentLoader
            Loader instance.

        """
        loader_cls = self.get_loader_class(name)
        return loader_cls(**kwargs)

    def get_loader_for_extension(self, extension: str) -> Optional[str]:
        """Get the name of a loader for a file extension.

        Parameters
        ----------
        extension : str
            File extension (with or without leading dot).

        Returns
        -------
        Optional[str]
            Name of the loader, or None if no loader is registered for the extension.

        """
        # Handle extensions with or without the leading dot
        ext = extension if extension.startswith(".") else f".{extension}"
        return self._extension_mapping.get(ext.lower())

    def create_loader_for_file(self, file_path: Union[str, Path], **kwargs) -> Optional[DocumentLoader]:
        """Create a loader instance for a file based on its extension.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the file.
        **kwargs
            Additional arguments to pass to the loader constructor.

        Returns
        -------
        Optional[DocumentLoader]
            Loader instance, or None if no loader is registered for the file's extension.

        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        loader_name = self.get_loader_for_extension(extension)
        if not loader_name:
            return None

        return self.create_loader(loader_name, **kwargs)


# Create a global registry instance
registry = LoaderRegistry()
