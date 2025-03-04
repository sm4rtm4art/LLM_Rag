#!/usr/bin/env python
"""Fixtures for integration tests."""

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_rag.rag.pipeline import RAGPipeline
from llm_rag.vectorstore.chroma import ChromaVectorStore, EmbeddingFunctionWrapper


class MockLLM(LLM):
    """Mock LLM for testing."""

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Return a mock response."""
        return "This is a mock response from the LLM."

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "mock"


@pytest.fixture
def vectorstore():
    """Create a test vector store with test documents."""
    # Create a temporary directory for the vector store
    temp_dir = tempfile.mkdtemp()

    # Create a test embedding model
    embeddings = EmbeddingFunctionWrapper(model_name="all-MiniLM-L6-v2")

    # Create a test vector store
    vector_store = ChromaVectorStore(
        persist_directory=temp_dir, collection_name="test_collection", embedding_function=embeddings
    )

    # Create test documents
    documents = [
        Document(page_content="This is a test document about AI.", metadata={"source": "test1.txt", "page": 1}),
        Document(page_content="This document discusses machine learning.", metadata={"source": "test2.txt", "page": 1}),
        Document(page_content="Python is a programming language.", metadata={"source": "test3.txt", "page": 1}),
    ]

    # Add documents to the vector store
    vector_store.add_documents(documents)

    yield vector_store

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def rag_pipeline(vectorstore):
    """Create a test RAG pipeline."""
    # Use a mock model for testing
    model = MockLLM()

    # Create a RAG pipeline
    pipeline = RAGPipeline(vectorstore=vectorstore, llm=model)

    return pipeline
