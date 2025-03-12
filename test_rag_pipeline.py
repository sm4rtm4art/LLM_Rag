#!/usr/bin/env python3
"""End-to-End Test of the RAG Pipeline

This script tests the complete RAG pipeline using the refactored document
loaders.
"""

import json
import logging
import tempfile
from pathlib import Path

# Import the new modular document loaders
from src.llm_rag.document_processing.loaders import (
    CSVLoader,
    JSONLoader,
    TextFileLoader,
    load_document,
    load_documents_from_directory,
)
from src.llm_rag.document_processing.processors import (
    DocumentProcessor,
    TextSplitter,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_documents():
    """Create test documents for the RAG pipeline."""
    temp_dir = tempfile.TemporaryDirectory()
    logger.info(f"Created temporary directory at {temp_dir.name}")

    # Create a few test files of different types

    # Text file
    text_file = Path(temp_dir.name) / "sample.txt"
    with open(text_file, "w") as f:
        f.write("""
        RAG (Retrieval-Augmented Generation) is a technique that enhances
        large language models by retrieving relevant information from
        external knowledge sources.

        It combines the strengths of retrieval-based systems and generative
        models to produce more accurate, up-to-date, and verifiable responses.

        The RAG architecture consists of three main components:
        1. The retriever, which searches for relevant documents
        2. The indexer, which stores and organizes the documents
        3. The generator, which produces the final response
        """)

    # CSV file
    csv_file = Path(temp_dir.name) / "data.csv"
    with open(csv_file, "w") as f:
        f.write("""topic,description
        Vector Databases,Vector databases are specialized database systems
        designed to store and search vector embeddings efficiently.
        Embeddings,Embeddings are numerical representations of data that
        capture semantic meaning in a high-dimensional space.
        Chunking Strategies,Effective chunking strategies are crucial for
        RAG systems to ensure relevant information is retrieved.
        """)

    # JSON file - Using json.dumps to avoid control characters and formatting issues
    json_file = Path(temp_dir.name) / "concepts.json"
    json_data = {
        "concepts": [
            {
                "name": "Cosine Similarity",
                "description": "A measure of similarity between two non-zero vectors "
                "that calculates the cosine of the angle between them.",
            },
            {
                "name": "Semantic Search",
                "description": "A search technique that understands the intent and "
                "contextual meaning of search queries rather than just "
                "matching keywords.",
            },
            {
                "name": "Knowledge Graph",
                "description": "A network of entities, their semantic types, properties, "
                "and relationships, used to enhance information retrieval "
                "with structured knowledge.",
            },
        ]
    }
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4)

    return temp_dir


def test_loading_individual_documents(temp_dir):
    """Test loading individual documents using specific loaders."""
    logger.info("Testing individual document loaders...")

    # Load text file
    text_file = Path(temp_dir.name) / "sample.txt"
    text_loader = TextFileLoader(text_file)
    text_docs = text_loader.load()
    logger.info(f"Loaded {len(text_docs)} documents from text file")
    logger.info(f"Text content sample: {text_docs[0]['content'][:50]}...")

    # Load CSV file
    csv_file = Path(temp_dir.name) / "data.csv"
    csv_loader = CSVLoader(csv_file, content_columns=["description"])
    csv_docs = csv_loader.load()
    logger.info(f"Loaded {len(csv_docs)} documents from CSV file")
    logger.info(f"CSV content sample: {csv_docs[0]['content'][:50]}...")

    # Load JSON file
    json_file = Path(temp_dir.name) / "concepts.json"
    json_loader = JSONLoader(json_file, jq_filter=".concepts[]", content_key="description")
    json_docs = json_loader.load()
    logger.info(f"Loaded {len(json_docs)} documents from JSON file")
    if json_docs:
        logger.info(f"JSON content sample: {json_docs[0]['content'][:50]}...")

    return text_docs + csv_docs + json_docs


def test_loading_with_factory(temp_dir):
    """Test loading documents using the factory functions."""
    logger.info("Testing factory loading functions...")

    # Load a specific file using the factory function
    text_file = Path(temp_dir.name) / "sample.txt"
    text_docs = load_document(text_file)
    logger.info(f"Factory loaded {len(text_docs)} documents from text file")

    # Load all documents from the directory
    all_docs = load_documents_from_directory(temp_dir.name)
    logger.info(f"Factory loaded {len(all_docs)} total documents from directory")

    return all_docs


def test_document_processing(documents):
    """Test document processing (chunking) on the loaded documents."""
    logger.info("Testing document processing...")

    # Create a text splitter for chunking
    # Using separators instead of chunker
    text_splitter = TextSplitter(chunk_size=200, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])

    # Create a document processor
    processor = DocumentProcessor(text_splitter)

    # Process the documents
    chunks = processor.process(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")

    # Show a sample chunk
    if chunks:
        logger.info(f"Sample chunk: {chunks[0]['content']}")

    return chunks


def main():
    """Run the end-to-end test of the RAG pipeline."""
    logger.info("Starting end-to-end test of the RAG pipeline...")

    # Create test documents
    temp_dir = create_test_documents()

    try:
        # Test individual loaders
        individual_docs = test_loading_individual_documents(temp_dir)

        # Test factory functions
        factory_docs = test_loading_with_factory(temp_dir)

        # Use either individual or factory loaded documents
        documents = factory_docs if factory_docs else individual_docs

        # Test document processing
        _ = test_document_processing(documents)

        logger.info("Document loading and processing completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error in end-to-end test: {e}")
        return False
    finally:
        # Clean up temporary directory
        temp_dir.cleanup()
        logger.info("Cleaned up temporary directory")


if __name__ == "__main__":
    main()
