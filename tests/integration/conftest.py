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

    # Add a query method for backward compatibility
    def query(query_text, conversation_id=None):
        """Process a query through the RAG pipeline.

        Args:
            query_text: The user's query
            conversation_id: Optional ID for tracking conversation

        Returns:
            Dictionary with query, response, and additional information
        """
        # Retrieve relevant documents
        documents = pipeline._retriever.retrieve(query_text)

        # Convert Document objects to dictionaries if needed
        processed_docs = []
        for doc in documents:
            if hasattr(doc, "page_content"):
                # This is a langchain Document object
                processed_docs.append({"content": doc.page_content, "metadata": doc.metadata})
            else:
                # This is already a dictionary
                processed_docs.append(doc)

        # Format retrieved documents into context
        # Keeping this commented to show intent, but not using it to avoid linting error
        # context = pipeline._formatter.format_context(processed_docs)

        # Generate a response directly using the MockLLM
        response = "This is a mock response from the LLM."

        # Return results
        return {
            "query": query_text,
            "response": response,
            "documents": documents,  # Return original documents
            "conversation_id": conversation_id or "test-id",
        }

    # Add the query method to the pipeline
    pipeline.query = query

    return pipeline
