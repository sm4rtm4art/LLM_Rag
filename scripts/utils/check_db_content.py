#!/usr/bin/env python
"""Check the content of the vector database.

This script checks the content of the vector database and prints information
about the documents stored in it.
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import our custom modules

import chromadb


def check_db_content(db_path, collection_name="test_collection", limit=5):
    """Check the content of a Chroma database.

    Args:
        db_path: Path to the Chroma database
        collection_name: Name of the collection to check
        limit: Number of documents to display

    Returns:
        None

    """
    try:
        print(f"Checking database at: {db_path}")
        client = chromadb.PersistentClient(db_path)

        # Get all collection names
        collections = client.list_collections()
        print(f"Collections in database: {collections}")

        # If no specific collection name provided, use the first one
        if not collection_name and collections:
            collection_name = collections[0]
            print(f"Using collection: {collection_name}")

        # Get the collection
        try:
            collection = client.get_collection(collection_name)

            # Get document count
            doc_count = collection.count()
            print(f"Number of documents: {doc_count}")

            # Get sample documents
            results = collection.get(limit=limit)

            # Print sample documents
            print("\nSample documents:")
            for i, (doc, metadata) in enumerate(
                zip(results["documents"], results["metadatas"], strict=False)
            ):
                print(f"\nDocument {i + 1}:")
                print(f"Metadata: {metadata}")
                print(f"Content preview: {doc[:300]}...")

            # Check for unique filenames
            all_metadatas = collection.get(limit=doc_count)["metadatas"]
            filenames = set()
            for metadata in all_metadatas:
                if "source" in metadata:
                    filenames.add(metadata["source"])
                if "filename" in metadata:
                    filenames.add(metadata["filename"])

            print(f"\nUnique filenames/sources: {sorted(filenames)}")
        except Exception as e:
            print(f"Error accessing collection: {e}")

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "test_chroma_db"

    collection_name = sys.argv[2] if len(sys.argv) > 2 else "test_collection"
    check_db_content(db_path, collection_name)

    # If synthetic_test_db exists, check it too
    if Path("synthetic_test_db").exists() and db_path != "synthetic_test_db":
        print("\n" + "=" * 50)
        print("Checking synthetic database:")
        check_db_content("synthetic_test_db", collection_name)
