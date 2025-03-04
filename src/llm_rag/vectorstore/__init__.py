"""Vector store module.

This module provides vector store implementations for storing and retrieving
document embeddings.
"""

from src.llm_rag.vectorstore.base import VectorStore
from src.llm_rag.vectorstore.chroma import ChromaRetriever, ChromaVectorStore, EmbeddingFunctionWrapper
from src.llm_rag.vectorstore.multimodal import (
    MultiModalEmbeddingFunction,
    MultiModalRetriever,
    MultiModalVectorStore,
)

__all__ = [
    "ChromaRetriever",
    "ChromaVectorStore",
    "EmbeddingFunctionWrapper",
    "MultiModalEmbeddingFunction",
    "MultiModalRetriever",
    "MultiModalVectorStore",
    "VectorStore",
]
