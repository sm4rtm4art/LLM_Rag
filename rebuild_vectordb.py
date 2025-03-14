#!/usr/bin/env python3
"""Integration script to rebuild the vector database.

This script demonstrates the proper integration flow with the loaders.py module:
1. Use the DirectoryLoader from loaders.py to load documents
2. Create embeddings for the documents
3. Store the documents and embeddings in a vector database
"""

import os
import shutil
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Important: Import directly from document_processing, not from loaders/
# This tests our backward compatibility layer
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.llm_rag.document_processing import DirectoryLoader
from src.llm_rag.vectorstore import ChromaVectorStore

# Constants
TEST_SUBSET_DIR = Path("data/documents/test_subset")
DB_PATH = Path("./rebuilt_test_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Small model for testing


def clean_database():
    """Remove existing database if it exists."""
    db_path = "rebuilt_test_db"
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}")
        shutil.rmtree(db_path)


def load_documents():
    """Load documents from the test_subset directory."""
    print("Loading documents from data/documents/test_subset")
    loader = DirectoryLoader("data/documents/test_subset")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def create_vector_store(documents):
    """Create a vector store from the documents."""
    print("Creating vector store")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = ChromaVectorStore(
        embedding_function=embedding_model, collection_name="test_collection", persist_directory="rebuilt_test_db"
    )

    # Add documents to the vector store
    vector_store.add_documents(documents)
    print(f"Added {len(documents)} documents to vector store")

    # Persist the vector store
    vector_store.persist()
    print("Vector store persisted successfully")

    return vector_store


def test_retrieval(vector_store):
    """List all documents in the collection."""
    print("\nListing all documents in the collection:")

    try:
        # Try to get all documents from the collection
        if hasattr(vector_store, "_collection"):
            # Access the Chroma collection directly
            collection = vector_store._collection
            results = collection.get()

            if results and "documents" in results and results["documents"]:
                documents = results["documents"]
                metadatas = results.get("metadatas", [None] * len(documents))

                print(f"Found {len(documents)} documents:")
                for i, (doc, meta) in enumerate(zip(documents, metadatas, strict=False)):
                    source = meta.get("source", "UNKNOWN") if meta else "UNKNOWN"
                    content_preview = doc[:100] + "..." if len(doc) > 100 else doc
                    print(f"\nDocument {i + 1}: {source}")
                    print(f"Preview: {content_preview}")
            else:
                print("No documents found in the collection.")
        else:
            print("Vector store does not support direct collection access.")
    except Exception as e:
        print(f"Error retrieving documents: {e}")


def main():
    """Rebuild the vector database."""
    print("Starting vector database rebuild process")

    # Step 1: Clean existing database
    clean_database()

    # Step 2: Load documents using DirectoryLoader from loaders.py
    documents = load_documents()

    # Step 3: Create vector store
    vector_store = create_vector_store(documents)

    # Step 4: Test retrieval
    test_retrieval(vector_store)

    print("\nVector database rebuild completed successfully")


if __name__ == "__main__":
    main()
