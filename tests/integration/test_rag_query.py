#!/usr/bin/env python
"""Test script for the RAG query functionality.

This script tests the RAG query functionality with a real LLM.
"""

import logging
import os
import pprint
import sys

# Add the project root to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Third-party imports
from src.llm_rag.models.factory import ModelBackend, ModelFactory
from src.llm_rag.rag.pipeline import RAGPipeline
from src.llm_rag.vectorstore.chroma import ChromaVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_rag_pipeline():
    """Set up the RAG pipeline for testing."""
    # Initialize the vector store
    vector_store = ChromaVectorStore(
        collection_name="rag_documents",
        persist_directory="./data/vectorstore",
    )

    # Initialize the LLM
    llm = ModelFactory.create_model(
        model_path_or_name="microsoft/phi-2",
        backend=ModelBackend.HUGGINGFACE,
        device="cpu",
        max_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    # Initialize the RAG pipeline
    pipeline = RAGPipeline(
        vectorstore=vector_store,
        llm=llm,
        top_k=5,
    )

    return pipeline


def test_single_query(rag_pipeline):
    """Test a single query."""
    query = "What are the requirements for fuses according to DIN VDE 0636-3?"
    logger.info(f"Query: {query}")
    result = rag_pipeline.query(query)

    logger.info(f"Response: {result['response']}")
    logger.info("Sources:")

    # Log the structure of each document
    for i, doc in enumerate(result["documents"]):
        logger.info(f"Document {i + 1} structure:")
        logger.info(pprint.pformat(doc))

        # Try to access the source from the document's metadata
        source = doc.get("metadata", {}).get("source", "Unknown")
        source_filename = doc.get("metadata", {}).get("filename", "")

        # Try to access the content from different possible fields
        content = None
        if "content" in doc and doc["content"]:
            content = doc["content"]
        elif "document" in doc and doc["document"]:
            content = doc["document"]
        elif "page_content" in doc and doc["page_content"]:
            content = doc["page_content"]

        if content:
            # Truncate content if it's too long
            if len(content) > 100:
                content = content[:100] + "..."

            # Log the document source and content
            if source_filename:
                logger.info(f"{i + 1}. {source}: {content}")
            else:
                logger.info(f"{i + 1}. {source}: {content}")

    logger.info("-" * 50)

    # Add an assertion instead of returning the result
    assert result is not None
    assert "response" in result
    assert "documents" in result


def main():
    """Run the test."""
    logger.info("Setting up RAG pipeline...")
    rag_pipeline = setup_rag_pipeline()
    logger.info("RAG pipeline setup complete")

    logger.info("Testing single query...")
    test_single_query(rag_pipeline)

    logger.info("Testing complete")


if __name__ == "__main__":
    main()
