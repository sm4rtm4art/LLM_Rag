#!/usr/bin/env python3
"""Test script to load documents from the test_subset directory.

This script demonstrates the use of the DirectoryLoader to load documents
from various file types in the test_subset directory.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the loader
from src.llm_rag.document_processing import DirectoryLoader, PDFLoader

# Define the path to the test_subset directory
TEST_SUBSET_DIR = Path("data/documents/test_subset")


def load_text_files():
    """Load text files from the test_subset directory."""
    print("\n--- Loading Text Files ---")

    try:
        # Create a loader for the test_subset directory
        loader = DirectoryLoader(
            directory_path=TEST_SUBSET_DIR,
            recursive=True,  # Also search subdirectories
            glob_pattern="*.txt",  # Start with just text files for simplicity
        )

        # Load the documents
        print("Loading text documents...")
        documents = loader.load()

        # Print summary
        print(f"\nSuccessfully loaded {len(documents)} text documents:")

        # Print details for each document
        for i, doc in enumerate(documents, 1):
            print(f"\nDocument {i}:")
            print(f"  Source: {doc['metadata'].get('source', 'Unknown')}")
            print(f"  Type: {doc['metadata'].get('filetype', 'Unknown')}")
            content_preview = doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"]
            print(f"  Content preview: {content_preview}")

    except Exception as e:
        print(f"Error loading text documents: {e}")
        return False

    return True


def load_pdf_files():
    """Load a PDF file from the test_subset directory."""
    print("\n--- Loading PDF Files ---")

    try:
        # Find a small PDF to load as an example
        pdf_path = next(TEST_SUBSET_DIR.glob("*.pdf"))
        print(f"Using PDF file: {pdf_path}")

        # Create a loader for this PDF file
        loader = PDFLoader(file_path=pdf_path)

        # Load the document
        print("Loading PDF document...")
        documents = loader.load()

        # Print summary
        print(f"\nSuccessfully loaded {len(documents)} pages from PDF:")

        # Print details for a few pages
        max_pages = min(3, len(documents))
        for i in range(max_pages):
            doc = documents[i]
            print(f"\nPage {i + 1}:")
            print(f"  Source: {doc['metadata'].get('source', 'Unknown')}")
            print(f"  Page: {doc['metadata'].get('page', 'Unknown')}")
            content_preview = doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"]
            print(f"  Content preview: {content_preview}")

    except Exception as e:
        print(f"Error loading PDF document: {e}")
        return False

    return True


def main():
    """Load documents from the test_subset directory and print summary."""
    print(f"Testing document loaders with files from {TEST_SUBSET_DIR}")

    # Test text file loading
    text_success = load_text_files()

    # Test PDF loading
    pdf_success = load_pdf_files()

    # Print overall result
    if text_success and pdf_success:
        print("\n✅ All document loading tests completed successfully!")
    else:
        print("\n❌ Some document loading tests failed.")


if __name__ == "__main__":
    main()
