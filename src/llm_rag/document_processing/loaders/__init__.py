"""Document loaders for the LLM-RAG system.

This module provides document loaders for different file types and sources.
All loaders implement the DocumentLoader interface and return standardized
documents.
"""

# Import the base classes and interfaces first
# Import the legacy loaders directly from the loaders.py file
# to avoid circular imports
import importlib.util
import sys
import warnings
from pathlib import Path

from .base import DirectoryLoader as BaseDirectoryLoader
from .base import DocumentLoader, FileLoader, LoaderRegistry, registry
from .base import WebLoader as BaseWebLoader

# Import the concrete loader implementations
from .directory_loader import DirectoryLoader

# Import factory functions
from .factory import get_available_loader_extensions, load_document, load_documents_from_directory
from .file_loaders import CSVLoader, TextFileLoader, XMLLoader
from .json_loader import JSONLoader
from .pdf_loaders import EnhancedPDFLoader, PDFLoader
from .web_loader import WebLoader, WebPageLoader

# Get the path to the legacy loaders.py file that's now in quarantine_backup
# Using absolute path resolution to make sure we find the file
project_root = Path(__file__).parent.parent.parent.parent.parent
loaders_path = project_root / "quarantine_backup" / "document_processing" / "loaders.py"
spec = importlib.util.spec_from_file_location("legacy_loaders", loaders_path)
legacy_loaders = importlib.util.module_from_spec(spec)
sys.modules["legacy_loaders"] = legacy_loaders
spec.loader.exec_module(legacy_loaders)


# Function to generate deprecation warning
def _deprecation_warning(old_class, new_class):
    warnings.warn(
        f"{old_class.__name__} is deprecated and will be removed in a "
        f"future version. Use {new_class.__name__} from "
        f"llm_rag.document_processing.loaders instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# Create wrapper classes for legacy loaders with deprecation warnings
class LegacyCSVLoader(legacy_loaders.CSVLoader):
    """Legacy CSV loader wrapper that emits deprecation warnings.

    DEPRECATED: Use CSVLoader from llm_rag.document_processing.loaders
    instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning.

        Parameters are passed to the underlying loader.
        """
        _deprecation_warning(legacy_loaders.CSVLoader, CSVLoader)
        super().__init__(*args, **kwargs)


class LegacyDirectoryLoader(legacy_loaders.DirectoryLoader):
    """Legacy directory loader wrapper that emits deprecation warnings.

    DEPRECATED: Use DirectoryLoader from llm_rag.document_processing.loaders
    instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning.

        Parameters are passed to the underlying loader.
        """
        _deprecation_warning(legacy_loaders.DirectoryLoader, DirectoryLoader)
        super().__init__(*args, **kwargs)


class LegacyDocumentLoader(legacy_loaders.DocumentLoader):
    """Legacy document loader wrapper that emits deprecation warnings.

    DEPRECATED: Use DocumentLoader from llm_rag.document_processing.loaders
    instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning.

        Parameters are passed to the underlying loader.
        """
        _deprecation_warning(legacy_loaders.DocumentLoader, DocumentLoader)
        super().__init__(*args, **kwargs)


class LegacyEnhancedPDFLoader(legacy_loaders.EnhancedPDFLoader):
    """Legacy enhanced PDF loader wrapper that emits deprecation warnings.

    DEPRECATED: Use EnhancedPDFLoader from llm_rag.document_processing.loaders
    instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning.

        Parameters are passed to the underlying loader.
        """
        _deprecation_warning(legacy_loaders.EnhancedPDFLoader, EnhancedPDFLoader)
        super().__init__(*args, **kwargs)


class LegacyJSONLoader(legacy_loaders.JSONLoader):
    """Legacy JSON loader wrapper that emits deprecation warnings.

    DEPRECATED: Use JSONLoader from llm_rag.document_processing.loaders
    instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning.

        Parameters are passed to the underlying loader.
        """
        _deprecation_warning(legacy_loaders.JSONLoader, JSONLoader)
        super().__init__(*args, **kwargs)


class LegacyPDFLoader(legacy_loaders.PDFLoader):
    """Legacy PDF loader wrapper that emits deprecation warnings.

    DEPRECATED: Use PDFLoader from llm_rag.document_processing.loaders
    instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning.

        Parameters are passed to the underlying loader.
        """
        _deprecation_warning(legacy_loaders.PDFLoader, PDFLoader)
        super().__init__(*args, **kwargs)


class LegacyTextFileLoader(legacy_loaders.TextFileLoader):
    """Legacy text file loader wrapper that emits deprecation warnings.

    DEPRECATED: Use TextFileLoader from llm_rag.document_processing.loaders
    instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning.

        Parameters are passed to the underlying loader.
        """
        _deprecation_warning(legacy_loaders.TextFileLoader, TextFileLoader)
        super().__init__(*args, **kwargs)


class LegacyWebPageLoader(legacy_loaders.WebPageLoader):
    """Legacy web page loader wrapper that emits deprecation warnings.

    DEPRECATED: Use WebPageLoader from llm_rag.document_processing.loaders
    instead.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with deprecation warning.

        Parameters are passed to the underlying loader.
        """
        _deprecation_warning(legacy_loaders.WebPageLoader, WebPageLoader)
        super().__init__(*args, **kwargs)


# For backward compatibility and public API access
__all__ = [
    # New modular base classes and interfaces
    "DocumentLoader",
    "FileLoader",
    "BaseDirectoryLoader",
    "BaseWebLoader",
    "LoaderRegistry",
    "registry",
    # New modular loader implementations
    "CSVLoader",
    "DirectoryLoader",
    "JSONLoader",
    "PDFLoader",
    "EnhancedPDFLoader",
    "TextFileLoader",
    "WebLoader",
    "WebPageLoader",
    "XMLLoader",
    # Factory functions
    "load_document",
    "load_documents_from_directory",
    "get_available_loader_extensions",
    # Legacy loaders (re-exported for backward compatibility)
    "LegacyCSVLoader",
    "LegacyDirectoryLoader",
    "LegacyDocumentLoader",
    "LegacyEnhancedPDFLoader",
    "LegacyJSONLoader",
    "LegacyPDFLoader",
    "LegacyTextFileLoader",
    "LegacyWebPageLoader",
]
