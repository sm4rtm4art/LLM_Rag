#!/usr/bin/env python
"""Simple script to demonstrate document loading functionality."""

from pathlib import Path

from llm_rag.document_processing.loaders import DirectoryLoader
from llm_rag.document_processing.processors import Documents


def load_documents(directory: str) -> Documents:
    """Load documents from a directory using the DirectoryLoader.

    Args:
        directory: Path to the directory containing documents

    Returns:
        List of loaded documents

    """
    # Create directory loader
    loader = DirectoryLoader(directory)

    # Load documents
    print(f"\nLoading documents from: {directory}")
    documents = loader.load()

    # Print summary
    print(f"\nLoaded {len(documents)} documents:")
    for i, doc in enumerate(documents, 1):
        metadata = doc.get("metadata", {})
        content = doc.get("content", "")
        source = metadata.get("source", "Unknown source")
        file_type = metadata.get("file_type", "Unknown")

        print(f"{i}. {source}")
        print(f"   Content length: {len(content)} characters")
        print(f"   File type: {file_type}")
        print()

    return documents


def main():
    """Load and process documents from a directory."""
    # Get the test data directory
    current_dir = Path(__file__).parent.parent
    test_data_dir = current_dir / "data" / "documents" / "test_subset"

    if not test_data_dir.exists():
        print(f"Error: Test data directory not found at {test_data_dir}")
        return

    # Load documents
    documents = load_documents(str(test_data_dir))

    # Print total statistics
    total_chars = sum(len(doc.get("content", "")) for doc in documents)
    print("\nTotal statistics:")
    print(f"Total documents: {len(documents)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average document length: {total_chars / len(documents):,.0f} characters")


if __name__ == "__main__":
    main()
