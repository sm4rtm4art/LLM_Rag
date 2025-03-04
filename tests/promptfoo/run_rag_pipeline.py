#!/usr/bin/env python3
"""
Script to run the RAG pipeline for promptfoo testing.

This script serves as a custom provider for promptfoo, taking a query as input,
running it through the RAG pipeline, and returning the response in a format
that promptfoo can process.
"""

import json
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Now we can import from the project
from src.llm_rag.pipeline import RAGPipeline  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_rag_pipeline(query, context=None):
    """
    Run the RAG pipeline with the given query.

    Args:
        query (str): The query to process
        context (dict, optional): Additional context (not used in this implementation)

    Returns:
        dict: A dictionary containing the response and metadata
    """
    logger.info(f"Running RAG pipeline with query: {query}")

    # Initialize the RAG pipeline
    pipeline = RAGPipeline()

    # Run the query
    response, metadata = pipeline.query(query)

    # Extract source documents
    sources = []
    if metadata and "source_documents" in metadata:
        for doc in metadata["source_documents"]:
            source = {
                "content": (doc.page_content if hasattr(doc, "page_content") else ""),
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
            }
            sources.append(source)

    # Format the result for promptfoo
    result = {"response": response, "sources": sources}

    return result


if __name__ == "__main__":
    # Check if a query was provided as a command-line argument
    if len(sys.argv) < 2:
        logger.error("No query provided. Usage: python run_rag_pipeline.py 'your query here'")
        sys.exit(1)

    # Get the query from command-line arguments
    query = sys.argv[1]

    # Run the pipeline
    result = run_rag_pipeline(query)

    # Print the result as JSON
    print(json.dumps(result, indent=2))
