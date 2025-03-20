"""Document loaders implementation modules.

This package contains the implementation of various document loaders.
The main entry point for using these loaders is through
the parent module: llm_rag.document_processing.loaders
"""

# Import this to make the registry available
from .base import LoaderRegistry, registry

# No direct re-exports to avoid namespace collision
# with llm_rag.document_processing.loaders

__all__ = [
    "registry",
    "LoaderRegistry",
]
