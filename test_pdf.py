#!/usr/bin/env python
"""Test script for PDF loader with the new pypdf library."""

from src.llm_rag.document_processing.loaders.pdf_loaders import PDFLoader


def main():
    """Test PDF loading with a real PDF file."""
    # Path to a test PDF file
    test_file = "data/documents/test_subset/VDE_0636-3_A2_E__DIN_VDE_0636-3_A2__2010-04.pdf"

    try:
        # Initialize the loader
        loader = PDFLoader(file_path=test_file)

        # Load the documents
        docs = loader.load()

        print(f"Successfully loaded {len(docs)} documents (pages) from PDF")

        # Print a preview of the first document
        if docs:
            print(f"First document content preview: {docs[0]['content'][:100]}...")
            print(f"Metadata: {docs[0]['metadata']}")

        print("PDF loader test completed successfully!")

    except Exception as e:
        print(f"Error loading PDF: {e}")


if __name__ == "__main__":
    main()
