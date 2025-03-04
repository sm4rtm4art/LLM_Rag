#!/usr/bin/env python3
"""Test script for document retrieval.

This script tests the retrieval capabilities of the RAG system
on the test files in data/documents/test_subset/.
"""

import logging
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
        from llm_rag.document_processing.loaders import DirectoryLoader  # noqa: E402

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
    """Process documents for retrieval testing."""
    logger.info("Processing documents")

    try:
        # Try to use RecursiveTextChunker
        from llm_rag.document_processing.chunking import RecursiveTextChunker  # noqa: E402

        chunker = RecursiveTextChunker(
            chunk_size=1000,
            chunk_overlap=200,
        )
        processed_docs = chunker.split_documents(documents)
        logger.info(f"Split into {len(processed_docs)} chunks")

        # Print sample of processed documents
        print("\n=== Sample of Processed Document Chunks ===")
        for i, doc in enumerate(processed_docs[:3]):  # Show first 3 chunks
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

        return processed_docs
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return []


def create_vectorstore(documents):
    """Create a vector store for testing retrieval."""
    logger.info("Creating vector store")

    try:
        # Prepare documents for vector store
        texts = []
        metadatas = []
        ids = []

        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Skip empty content
            if not content or content.strip() == "":
                logger.warning(f"Skipping empty content from {metadata.get('source', 'unknown')}")
                continue

            texts.append(content)
            metadatas.append(metadata)
            ids.append(f"doc_{i}")

        # Create vector store
        from llm_rag.vectorstore.chroma import ChromaVectorStore  # noqa: E402

        vector_store = ChromaVectorStore(
            collection_name="test_retrieval",
            persist_directory="test_chroma_db",
        )

        # Add documents to vector store
        vector_store.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Added {len(texts)} documents to vector store")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None


def test_retrieval(vectorstore):
    """Test document retrieval."""
    logger.info("Testing retrieval")

    test_queries = [
        "What is RAG?",
        "How does retrieval augmented generation work?",
        "What are embeddings?",
        "Explain vector databases",
    ]

    for query in test_queries:
        print(f"\n\n=== Query: {query} ===")
        try:
            # Retrieve documents
            retrieved_docs = vectorstore.similarity_search(query, k=3)
            print(f"Retrieved {len(retrieved_docs)} documents")

            # Print retrieved documents
            for i, doc in enumerate(retrieved_docs):
                # Handle Document objects correctly
                # The Document object might have page_content and metadata attributes
                # instead of being a dict with get method
                if hasattr(doc, "page_content"):
                    content = doc.page_content
                    metadata = doc.metadata if hasattr(doc, "metadata") else {}
                else:
                    # Fallback to dictionary-like access
                    content = doc.get("content", "") if hasattr(doc, "get") else str(doc)
                    metadata = doc.get("metadata", {}) if hasattr(doc, "get") else {}

                # Get source file name
                source = metadata.get("source", "") if hasattr(metadata, "get") else str(metadata)
                source_file = Path(source).name if source else "unknown"

                # Print header with source info
                print(f"\n[{i + 1}] {source_file}")

                # Print content (truncate if too long)
                content_preview = content[:500] + "..." if len(content) > 500 else content
                print(f"\nContent: {content_preview}")
                print("-" * 80)
        except Exception as e:
            logger.error(f"Error retrieving documents for query '{query}': {e}")


def main():
    """Run the retrieval test."""
    logger.info("Starting retrieval test")

    # Load documents
    documents = load_documents()
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return

    # Process documents
    processed_docs = process_documents(documents)
    if not processed_docs:
        logger.error("No processed documents. Exiting.")
        return

    # Create vector store
    vector_store = create_vectorstore(processed_docs)
    if not vector_store:
        logger.error("Failed to create vector store. Exiting.")
        return

    # Test retrieval
    test_retrieval(vector_store)
    logger.info("Retrieval test completed")


if __name__ == "__main__":
    main()
