"""LLM RAG System.

This package implements a Retrieval Augmented Generation (RAG) system using:
- Vector stores for efficient document retrieval
- Embedding models for semantic search
- FastAPI for serving endpoints

Main Components:
- vectorstore: Document storage and retrieval
- api: REST API endpoints
"""

from llm_rag.version import __version__

# Re-export version
__version__ = __version__
__author__ = "Martin"
__email__ = "your.email@example.com"
