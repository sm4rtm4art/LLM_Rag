"""LLM RAG project.

This is a wrapper module to make imports in tests work correctly.
It forwards imports to the actual implementation in src.llm_rag.
"""

from src.llm_rag import *  # noqa

from llm_rag.version import __version__

# Re-export version
__version__ = __version__
