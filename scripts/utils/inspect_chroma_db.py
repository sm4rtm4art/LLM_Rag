#!/usr/bin/env python
"""Inspect a Chroma database.

This script inspects a Chroma database and prints information about the
collections and documents stored in it.
"""

import logging
import os
import sys

# Add the project root to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import our custom modules

from langchain_community.vectorstores import Chroma


def inspect_chroma_db(persist_directory, collection_name):
    """Inspect the contents of a Chroma database."""
    try:
        db = Chroma(persist_directory=persist_directory, collection_name=collection_name)
        count = db._collection.count()
        print(f"Number of documents: {count}")

        if count > 0:
            print("\nSample documents:")
            results = db.get(limit=min(10, count))

            for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"], strict=False)):
                print(f"\n--- Document {i + 1} ---")
                print(f"Metadata: {metadata}")
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                print(f"Content: {preview}")

            # Print unique filenames
            filenames = set()
            all_results = db.get(limit=count)
            for metadata in all_results["metadatas"]:
                if "filename" in metadata:
                    filenames.add(metadata["filename"])

            print("\n--- Unique Filenames ---")
            for filename in sorted(filenames):
                print(filename)

        return True
    except Exception as e:
        print(f"Error inspecting Chroma DB: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    inspect_chroma_db("test_chroma_db", "test_collection")
