#!/usr/bin/env python
"""Demo script for RAG (Retrieval-Augmented Generation) queries.

This script demonstrates how to use the RAG system with the HuggingFace LLM
and the documents loaded into the vector store.
"""

import argparse
import logging
import sys

from src.llm_rag.models.factory import ModelBackend, ModelFactory
from src.llm_rag.rag.pipeline import RAGPipeline
from src.llm_rag.vectorstore.chroma import ChromaVectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the script.

    Returns
    -------
        argparse.ArgumentParser: The configured argument parser.

    """
    parser = argparse.ArgumentParser(description="Demo script for RAG queries with HuggingFace LLM.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/phi-2",
        help="HuggingFace model to use",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="documents",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="chroma_db",
        help="Path to ChromaDB database",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to ask the RAG system",
    )
    return parser


def run_rag_query(
    query: str,
    model_name: str,
    db_path: str,
    collection_name: str,
) -> str:
    """Run a RAG query against the system.

    Args:
    ----
        query: The query to ask.
        model_name: The HuggingFace model to use.
        db_path: Path to the ChromaDB database.
        collection_name: Name of the ChromaDB collection.

    Returns:
    -------
        The RAG system's response.

    """
    logger.info(f"Loading LLM model: {model_name}")

    # Create the LLM
    factory = ModelFactory()
    llm = factory.create_model(
        model_path_or_name=model_name,
        backend=ModelBackend.HUGGINGFACE,
        device="mps",  # Use MPS for Mac with Apple Silicon, or "cuda" for NVIDIA GPUs
    )

    # Load the vector store
    logger.info(f"Loading vector store from {db_path}, collection {collection_name}")
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=db_path,
    )

    # Create the RAG pipeline
    logger.info("Creating RAG pipeline")
    rag_pipeline = RAGPipeline(
        vectorstore=vector_store,
        llm=llm,
        top_k=3,  # Number of documents to retrieve
    )

    # Run the query
    logger.info(f"Running query: {query}")
    response = rag_pipeline.answer(query)

    return response


def main() -> None:
    """Run the main script logic."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    if not args.query:
        query = input("Enter your query: ")
    else:
        query = args.query

    try:
        response = run_rag_query(
            query=query,
            model_name=args.model_name,
            db_path=args.db_path,
            collection_name=args.collection,
        )

        print("\nRAG System Response:")
        print("-------------------")
        print(response)
        print("-------------------")

    except Exception as e:
        logger.error(f"Error running RAG query: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
