#!/usr/bin/env python
"""Simple script to load documents into the vector store.

This script uses only the available classes in src.llm_rag to load documents
into the vector store.
"""

import logging
import os
import sys

# Add the current directory to the path so we can import the src module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.llm_rag.document_processing.chunking import RecursiveTextChunker
from src.llm_rag.document_processing.loaders import DirectoryLoader
from src.llm_rag.vectorstore.chroma import ChromaVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Load documents into the vector store."""
    # Parameters
    dir_path = "data/documents/test_subset"
    glob_pattern = "**/*.pdf"
    db_path = "chroma_db"
    collection_name = "documents"
    chunk_size = 1000
    chunk_overlap = 200

    # Check if the document directory exists
    if not os.path.exists(dir_path):
        logger.error(f"Document directory not found: {dir_path}")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)

    # Load documents
    logger.info(f"Loading documents from {dir_path} with pattern {glob_pattern}")
    loader = DirectoryLoader(
        directory_path=dir_path,
        recursive=True,
        glob_pattern=glob_pattern,
    )

    try:
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        sys.exit(1)

    if not documents:
        logger.error("No documents were loaded. Check your input directory and glob pattern.")
        sys.exit(1)

    # Split documents into chunks
    logger.info(f"Splitting {len(documents)} documents into chunks")
    chunker = RecursiveTextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    document_chunks = chunker.split_documents(documents)
    logger.info(f"Created {len(document_chunks)} chunks")

    if not document_chunks:
        logger.error("No document chunks were created. Check your input documents.")
        sys.exit(1)

    # Prepare documents for vector store
    texts = []
    metadatas = []
    ids = []

    for i, doc in enumerate(document_chunks):
        # Extract content and metadata
        if isinstance(doc, dict) and "content" in doc and "metadata" in doc:
            texts.append(doc["content"])
            metadatas.append(doc["metadata"])
        else:
            # Handle unexpected format
            logger.warning(f"Unexpected document format: {type(doc)}")
            continue

        # Generate ID
        doc_id = f"doc_{i}"
        ids.append(doc_id)

    logger.info(f"Prepared {len(texts)} documents for vector store")

    if not texts:
        logger.error("Failed to prepare documents for vector store.")
        sys.exit(1)

    # Create vector store
    logger.info(f"Creating vector store at {db_path} with collection {collection_name}")
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=db_path,
    )

    # Add documents to vector store
    logger.info("Adding documents to vector store...")
    vector_store.collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info(f"Successfully added {len(texts)} document chunks to the vector store")
    logger.info(f"Vector store path: {db_path}")
    logger.info(f"Collection name: {collection_name}")
    logger.info("You can now use the RAG system with the loaded documents")


if __name__ == "__main__":
    main()
