#!/usr/bin/env python
"""Check the content of the vector database.

This script checks the content of the vector database and prints information
about the documents stored in it.
"""

import logging
import os
import sys
from pathlib import Path

import chromadb

# Add the project root to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_db_content(db_path, collection_name="test_collection", limit=5):
    """Check the content of a Chroma database.

    Args:
        db_path: Path to the Chroma database
        collection_name: Name of the collection to check
        limit: Maximum number of documents to display

    Returns:
        None

    """
    # Check if the database exists
    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}")
        return

    # Create a client
    client = chromadb.PersistentClient(path=db_path)

    # Get the collection
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        logger.error(f"Collection {collection_name} not found in database")
        return

    # Get all documents
    results = collection.get(limit=limit)

    # Print information about the collection
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Number of documents: {collection.count()}")

    # Print sample documents
    logger.info("Sample documents:")
    for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"], strict=False)):
        logger.info(f"Document {i}:")
        logger.info(f"  Content: {doc[:100]}...")
        logger.info(f"  Metadata: {metadata}")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Check the content of a Chroma database")
    parser.add_argument(
        "--db-path",
        type=str,
        default="test_chroma_db",
        help="Path to the Chroma database",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="test_collection",
        help="Name of the collection to check",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of documents to display",
    )
    args = parser.parse_args()

    # Check the database content
    check_db_content(args.db_path, args.collection, args.limit)

    # If synthetic_test_db exists, check it too
    if Path("synthetic_test_db").exists() and args.db_path != "synthetic_test_db":
        print("\n" + "=" * 50)
        print("Checking synthetic database:")
        check_db_content("synthetic_test_db", args.collection, args.limit)
