"""Vector store implementations for document storage and retrieval.

This module provides:
- Base VectorStore interface
- ChromaDB implementation
- Embedding function wrappers

The vector store is responsible for:
1. Storing document embeddings
2. Performing semantic similarity search
3. Managing document metadata
"""

from .base import VectorStore
from .chroma import ChromaVectorStore, EmbeddingFunctionWrapper

__all__ = ["VectorStore", "ChromaVectorStore", "EmbeddingFunctionWrapper"]
