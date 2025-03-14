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

# Set default for legacy loaders
_has_legacy_loaders = False
legacy_loaders = None

# Try to load legacy loaders, but allow it to fail gracefully in CI environments
try:
    # Using absolute path resolution to make sure we find the file
    project_root = Path(__file__).parent.parent.parent.parent.parent
    loaders_path = project_root / "quarantine_backup" / "document_processing" / "loaders.py"

    if loaders_path.exists():
        # Only attempt to load if the file exists
        spec = importlib.util.spec_from_file_location("legacy_loaders", loaders_path)
        legacy_loaders = importlib.util.module_from_spec(spec)
        sys.modules["legacy_loaders"] = legacy_loaders
        spec.loader.exec_module(legacy_loaders)
        _has_legacy_loaders = True
    else:
        warnings.warn(
            "Legacy loaders.py file not found. Legacy loader functionality will be disabled.",
            ImportWarning,
            stacklevel=2,
        )
except Exception as e:
    warnings.warn(
        f"Failed to import legacy loaders: {str(e)}. Legacy loader functionality will be disabled.",
        ImportWarning,
        stacklevel=2,
    )


# Function to generate deprecation warning
def _deprecation_warning(old_class, new_class):
    """Generate a deprecation warning for legacy loaders."""
    warnings.warn(
        f"{old_class.__name__} is deprecated and will be removed in a "
        f"future version. Use {new_class.__name__} from "
        f"llm_rag.document_processing.loaders instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# Base exception for legacy loader not available
class LegacyLoaderNotAvailableError(NotImplementedError):
    """Exception raised when legacy loaders are requested but not available."""

    def __init__(self, loader_name):
        """Initialize with loader name."""
        super().__init__(
            f"{loader_name} is not available because the legacy loaders.py file "
            f"is not present. Please use the new loaders from "
            f"llm_rag.document_processing.loaders instead."
        )


# Define legacy loader classes based on availability
if _has_legacy_loaders:
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
else:
    # Create placeholder classes for all legacy loaders
    class LegacyCSVLoader:
        """Legacy CSV loader placeholder that raises error when legacy file is not available."""

        def __init__(self, *args, **kwargs):
            """Raise error as the legacy implementation is not available."""
            raise LegacyLoaderNotAvailableError("LegacyCSVLoader")

    class LegacyDirectoryLoader:
        """Legacy directory loader placeholder that raises error when legacy file is not available."""

        def __init__(self, *args, **kwargs):
            """Raise error as the legacy implementation is not available."""
            raise LegacyLoaderNotAvailableError("LegacyDirectoryLoader")

    class LegacyDocumentLoader:
        """Legacy document loader placeholder that raises error when legacy file is not available."""

        def __init__(self, *args, **kwargs):
            """Raise error as the legacy implementation is not available."""
            raise LegacyLoaderNotAvailableError("LegacyDocumentLoader")

    class LegacyEnhancedPDFLoader:
        """Legacy enhanced PDF loader placeholder that raises error when legacy file is not available."""

        def __init__(self, *args, **kwargs):
            """Raise error as the legacy implementation is not available."""
            raise LegacyLoaderNotAvailableError("LegacyEnhancedPDFLoader")

    class LegacyJSONLoader:
        """Legacy JSON loader placeholder that raises error when legacy file is not available."""

        def __init__(self, *args, **kwargs):
            """Raise error as the legacy implementation is not available."""
            raise LegacyLoaderNotAvailableError("LegacyJSONLoader")

    class LegacyPDFLoader:
        """Legacy PDF loader placeholder that raises error when legacy file is not available."""

        def __init__(self, *args, **kwargs):
            """Raise error as the legacy implementation is not available."""
            raise LegacyLoaderNotAvailableError("LegacyPDFLoader")

    class LegacyTextFileLoader:
        """Legacy text file loader placeholder that raises error when legacy file is not available."""

        def __init__(self, *args, **kwargs):
            """Raise error as the legacy implementation is not available."""
            raise LegacyLoaderNotAvailableError("LegacyTextFileLoader")

    class LegacyWebPageLoader:
        """Legacy web page loader placeholder that raises error when legacy file is not available."""

        def __init__(self, *args, **kwargs):
            """Raise error as the legacy implementation is not available."""
            raise LegacyLoaderNotAvailableError("LegacyWebPageLoader")


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
