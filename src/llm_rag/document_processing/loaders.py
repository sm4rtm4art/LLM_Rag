"""Document loaders for the LLM-RAG system.

This module provides components for loading documents from various sources
and formats, including PDFs, text files, CSV files, JSON, and web content.

Note: This file is maintained for backward compatibility. For new development,
please use the modular components in the loaders/ directory.
"""

import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    # First try to import from the modular implementation
    from .loaders.base import DocumentLoader
    from .loaders.directory_loader import DirectoryLoader as ModularDirectoryLoader
    from .loaders.file_loaders import CSVLoader as ModularCSVLoader
    from .loaders.file_loaders import EnhancedPDFLoader as ModularEnhancedPDFLoader
    from .loaders.file_loaders import JSONLoader as ModularJSONLoader
    from .loaders.file_loaders import PDFLoader as ModularPDFLoader
    from .loaders.file_loaders import TextFileLoader as ModularTextFileLoader
    from .loaders.file_loaders import XMLLoader as ModularXMLLoader
    from .loaders.web_loaders import WebLoader as ModularWebLoader
    from .loaders.web_loaders import WebPageLoader as ModularWebPageLoader

    # Import successful, set up compatibility layer
    HAS_MODULAR_LOADERS = True

    # Issue deprecation warning
    warnings.warn(
        "The monolithic loaders.py module is deprecated and will be removed in a future version. "
        "Please use the modular loaders from llm_rag.document_processing.loaders instead.",
        DeprecationWarning,
        stacklevel=2,
    )
except ImportError as e:
    # If import fails, use stub classes
    warnings.warn(
        f"Failed to import modular document loader components: {e}. Using stub classes.",
        stacklevel=2,
    )
    _MODULAR_IMPORT_SUCCESS = False

# Get configured logger
logger = logging.getLogger(__name__)

# Type aliases for documents
Documents = List[Dict[str, Any]]


# Define base Document Loader class
class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self) -> Documents:
        """Load documents from a source.

        Returns
        -------
            List of documents, where each document is a dictionary with
            'content' and 'metadata' keys.

        """
        pass


if _MODULAR_IMPORT_SUCCESS:
    # Import was successful, use the new implementations with adapters

    class TextFileLoader(ModularTextFileLoader):
        """Adapter for backward compatibility with TextFileLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize the TextFileLoader.

            Args:
                file_path: Path to the text file
                **kwargs: Additional arguments

            """
            super().__init__(file_path=file_path, **kwargs)

    class PDFLoader(ModularPDFLoader):
        """Adapter for backward compatibility with PDFLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize the PDFLoader.

            Args:
                file_path: Path to the PDF file
                **kwargs: Additional arguments

            """
            super().__init__(file_path=file_path, **kwargs)

    class EnhancedPDFLoader(ModularEnhancedPDFLoader):
        """Adapter for backward compatibility with EnhancedPDFLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize the EnhancedPDFLoader.

            Args:
                file_path: Path to the PDF file
                **kwargs: Additional arguments

            """
            super().__init__(file_path=file_path, **kwargs)

    class CSVLoader(ModularCSVLoader):
        """Adapter for backward compatibility with CSVLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize the CSVLoader.

            Args:
                file_path: Path to the CSV file
                **kwargs: Additional arguments

            """
            super().__init__(file_path=file_path, **kwargs)

    class JSONLoader(ModularJSONLoader):
        """Adapter for backward compatibility with JSONLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize the JSONLoader.

            Args:
                file_path: Path to the JSON file
                **kwargs: Additional arguments

            """
            super().__init__(file_path=file_path, **kwargs)

    class WebLoader(ModularWebLoader):
        """Adapter for backward compatibility with WebLoader."""

        def __init__(self, web_path: str, **kwargs):
            """Initialize the WebLoader.

            Args:
                web_path: URL to load
                **kwargs: Additional arguments

            """
            super().__init__(web_path=web_path, **kwargs)

    class WebPageLoader(ModularWebPageLoader):
        """Adapter for backward compatibility with WebPageLoader."""

        def __init__(self, web_path: str, **kwargs):
            """Initialize the WebPageLoader.

            Args:
                web_path: URL to load
                **kwargs: Additional arguments

            """
            super().__init__(web_path=web_path, **kwargs)

    class DirectoryLoader(ModularDirectoryLoader):
        """Adapter for backward compatibility with DirectoryLoader."""

        def __init__(
            self,
            directory_path: Union[str, Path],
            glob_pattern: str = "*.*",
            recursive: bool = True,
            **kwargs,
        ):
            """Initialize the DirectoryLoader.

            Args:
                directory_path: Path to the directory
                glob_pattern: Pattern to match files
                recursive: Whether to search recursively
                **kwargs: Additional arguments

            """
            super().__init__(
                directory_path=directory_path,
                glob_pattern=glob_pattern,
                recursive=recursive,
                **kwargs,
            )

    class XMLLoader(ModularXMLLoader):
        """Adapter for backward compatibility with XMLLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize the XMLLoader.

            Args:
                file_path: Path to the XML file
                **kwargs: Additional arguments

            """
            super().__init__(file_path=file_path, **kwargs)

else:
    # Import failed, use stub classes
    class TextFileLoader(DocumentLoader):
        """Stub class for TextFileLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize stub TextFileLoader.

            Args:
                file_path: Path to the text file
                **kwargs: Additional arguments

            """
            pass

        def load(self) -> Documents:
            """Load documents from text file.

            Returns:
                Empty list of documents

            """
            warnings.warn("Using stub TextFileLoader. Install required dependencies.", stacklevel=2)
            return []

    class PDFLoader(DocumentLoader):
        """Stub class for PDFLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize stub PDFLoader.

            Args:
                file_path: Path to the PDF file
                **kwargs: Additional arguments

            """
            pass

        def load(self) -> Documents:
            """Load documents from PDF file.

            Returns:
                Empty list of documents

            """
            warnings.warn("Using stub PDFLoader. Install required dependencies.", stacklevel=2)
            return []

    class EnhancedPDFLoader(DocumentLoader):
        """Stub class for EnhancedPDFLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize stub EnhancedPDFLoader.

            Args:
                file_path: Path to the PDF file
                **kwargs: Additional arguments

            """
            pass

        def load(self) -> Documents:
            """Load documents from PDF file with enhanced processing.

            Returns:
                Empty list of documents

            """
            warnings.warn("Using stub EnhancedPDFLoader. Install required dependencies.", stacklevel=2)
            return []

    class CSVLoader(DocumentLoader):
        """Stub class for CSVLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize stub CSVLoader.

            Args:
                file_path: Path to the CSV file
                **kwargs: Additional arguments

            """
            pass

        def load(self) -> Documents:
            """Load documents from CSV file.

            Returns:
                Empty list of documents

            """
            warnings.warn("Using stub CSVLoader. Install required dependencies.", stacklevel=2)
            return []

    class JSONLoader(DocumentLoader):
        """Stub class for JSONLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize stub JSONLoader.

            Args:
                file_path: Path to the JSON file
                **kwargs: Additional arguments

            """
            pass

        def load(self) -> Documents:
            """Load documents from JSON file.

            Returns:
                Empty list of documents

            """
            warnings.warn("Using stub JSONLoader. Install required dependencies.", stacklevel=2)
            return []

    class WebLoader(DocumentLoader):
        """Stub class for WebLoader."""

        def __init__(self, web_path: str, **kwargs):
            """Initialize stub WebLoader.

            Args:
                web_path: URL to load
                **kwargs: Additional arguments

            """
            pass

        def load(self) -> Documents:
            """Load documents from web URL.

            Returns:
                Empty list of documents

            """
            warnings.warn("Using stub WebLoader. Install required dependencies.", stacklevel=2)
            return []

    class WebPageLoader(DocumentLoader):
        """Stub class for WebPageLoader."""

        def __init__(self, web_path: str, **kwargs):
            """Initialize stub WebPageLoader.

            Args:
                web_path: URL to load
                **kwargs: Additional arguments

            """
            pass

        def load(self) -> Documents:
            """Load documents from web page.

            Returns:
                Empty list of documents

            """
            warnings.warn("Using stub WebPageLoader. Install required dependencies.", stacklevel=2)
            return []

    class DirectoryLoader(DocumentLoader):
        """Stub class for DirectoryLoader."""

        def __init__(
            self, directory_path: Union[str, Path], glob_pattern: str = "**/*", recursive: bool = True, **kwargs
        ):
            """Initialize stub DirectoryLoader.

            Args:
                directory_path: Path to the directory
                glob_pattern: Pattern to match files
                recursive: Whether to search recursively
                **kwargs: Additional arguments

            """
            pass

        def load(self) -> Documents:
            """Load documents from directory.

            Returns:
                Empty list of documents

            """
            warnings.warn("Using stub DirectoryLoader. Install required dependencies.", stacklevel=2)
            return []

    class XMLLoader(DocumentLoader):
        """Stub class for XMLLoader."""

        def __init__(self, file_path: Union[str, Path], **kwargs):
            """Initialize stub XMLLoader.

            Args:
                file_path: Path to the XML file
                **kwargs: Additional arguments

            """
            pass

        def load(self) -> Documents:
            """Load documents from XML file.

            Returns:
                Empty list of documents

            """
            warnings.warn("Using stub XMLLoader. Install required dependencies.", stacklevel=2)
            return []


# Suppress deprecation warnings for backward compatibility
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
