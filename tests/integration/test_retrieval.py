#!/usr/bin/env python3
"""Test script for document retrieval.

This script tests the retrieval capabilities of the RAG system
on the test files in data/documents/test_subset/.
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Test directory
TEST_DIR = Path("data/documents/test_subset")


def load_documents():
    """Load documents from the test directory."""
    logger.info(f"Loading documents from: {TEST_DIR}")

    try:
        from llm_rag.document_processing.loaders import DirectoryLoader

        loader = DirectoryLoader(
            directory_path=TEST_DIR,
            recursive=True,
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document chunks")

        # Count document types
        doc_types: dict[str, int] = {}
        for doc in documents:
            filetype = doc.get("metadata", {}).get("filetype", "unknown")
            doc_types[filetype] = doc_types.get(filetype, 0) + 1

        logger.info(f"Document types: {doc_types}")

        # Print sample of loaded documents
        print("\n=== Sample of Loaded Documents ===")
        for i, doc in enumerate(documents[:3]):  # Show first 3 documents
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Get source file name
            source = metadata.get("source", "unknown")
            source_file = Path(source).name if source else "unknown"

            # Get content type
            filetype = metadata.get("filetype", "unknown").upper()

            # Print header with source info
            print(f"\n[{i + 1}] {filetype} - {source_file}")

            # Print metadata
            print(f"Page: {metadata.get('page_num', 'N/A')}")
            if "section_title" in metadata:
                print(f"Section: {metadata.get('section_title', '')}")

            # Print content (truncate if too long)
            content_preview = content[:200] + "..." if len(content) > 200 else content
            print(f"\nContent: {content_preview}")
            print("-" * 80)

        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []


def process_documents(documents):
    """Process and chunk documents."""
    logger.info("Processing documents")

    try:
        # Try to use RecursiveTextChunker
        from llm_rag.document_processing.chunking import RecursiveTextChunker

        chunker = RecursiveTextChunker(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunked_docs = chunker.split_documents(documents)
        logger.info(f"Created {len(chunked_docs)} chunks")

        # Print sample of chunked documents
        print("\n=== Sample of Chunked Documents ===")
        for i, doc in enumerate(chunked_docs[:3]):  # Show first 3 chunks
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Get source file name
            source = metadata.get("source", "unknown")
            source_file = Path(source).name if source else "unknown"

            # Get content type
            filetype = metadata.get("filetype", "unknown").upper()

            # Print header with source info
            print(f"\n[{i + 1}] {filetype} - {source_file}")

            # Print metadata
            print(f"Page: {metadata.get('page_num', 'N/A')}")
            if "section_title" in metadata:
                print(f"Section: {metadata.get('section_title', '')}")

            # Print content (truncate if too long)
            content_preview = content[:200] + "..." if len(content) > 200 else content
            print(f"\nContent: {content_preview}")
            print("-" * 80)

        return chunked_docs
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return []


def create_vectorstore(documents):
    """Create a vector store from processed documents."""
    logger.info("Creating vector store")

    try:
        # Use ChromaVectorStore
        from src.llm_rag.vectorstore.chroma import ChromaVectorStore

        vectorstore = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory="synthetic_test_db",
        )

        # Extract content and metadata
        texts = []
        metadatas = []
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Skip empty content
            if not content or content.strip() == "":
                logger.warning(f"Skipping empty content from {metadata.get('source', 'unknown')}")
                continue

            texts.append(content)
            metadatas.append(metadata)

        # Add documents to vector store
        if texts:
            vectorstore.add_documents(texts, metadatas)
            logger.info(f"Added {len(texts)} documents to vector store")
        else:
            logger.warning("No documents with content to add to vector store")

        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None


def test_retrieval(vectorstore):
    """Test retrieval from the vector store."""
    if not vectorstore:
        logger.error("Vector store not available")
        return

    # Test queries
    test_queries = [
        "What is a RAG system?",
        "Tell me about LLaMA 3",
        "What are DIN standards?",
        "Explain safety requirements in DIN standards",
    ]

    for query in test_queries:
        logger.info(f"Testing query: {query}")
        try:
            results = vectorstore.search(query, n_results=3)
            print(f"\n=== Results for: {query} ===")

            for i, result in enumerate(results):
                # The search method returns 'document' not 'content'
                content = result.get("document", "")
                metadata = result.get("metadata", {})

                # Get source file name
                source = metadata.get("source", "unknown")
                source_file = Path(source).name if source else "unknown"

                # Get content type
                filetype = metadata.get("filetype", "unknown").upper()

                # Print header with source info
                print(f"\n[{i + 1}] {filetype} - {source_file}")

                # Print metadata
                print(f"Page: {metadata.get('page_num', 'N/A')}")
                if "section_title" in metadata:
                    print(f"Section: {metadata.get('section_title', '')}")

                # Print content (truncate if too long)
                content_preview = content[:500] + "..." if len(content) > 500 else content
                print(f"\nContent: {content_preview}")
                print("-" * 80)
        except Exception as e:
            logger.error(f"Error testing retrieval: {e}")


def main():
    """Run the retrieval test."""
    # Load documents
    documents = load_documents()
    if not documents:
        logger.error("No documents loaded")
        return

    # Process documents
    processed_docs = process_documents(documents)
    if not processed_docs:
        logger.error("No processed documents")
        return

    # Create vector store
    vectorstore = create_vectorstore(processed_docs)
    if not vectorstore:
        logger.error("Vector store creation failed")
        return

    # Test retrieval
    test_retrieval(vectorstore)


if __name__ == "__main__":
    main()
