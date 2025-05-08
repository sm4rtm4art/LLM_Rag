#!/usr/bin/env python3
"""Check the content of the vector store and search for DIN information.

This script provides utilities to inspect the contents of a ChromaDB
vector store, retrieve documents, and perform similarity searches.
"""

import logging
import os
import sys
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add the project root to the path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

try:
    from src.llm_rag.vectorstore.chroma import ChromaVectorStore
except ImportError:
    logger.error('Failed to import ChromaVectorStore. Make sure the module is installed.')
    sys.exit(1)


def check_vectorstore(collection_name: str = 'documents', persist_directory: str = 'chroma_db') -> None:
    """Check the content of the vector store.

    Args:
        collection_name: Name of the collection in ChromaDB
        persist_directory: Directory where ChromaDB data is stored

    """
    try:
        # Initialize the vector store
        vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        # Get all documents
        docs = vector_store.get_all_documents()
        logger.info(f'Number of documents in vector store: {len(docs)}')

        # Print sample documents
        logger.info('Sample document content:')
        for i, doc in enumerate(docs[:3]):
            # Document objects have page_content and metadata attributes
            content = getattr(doc, 'page_content', '')
            metadata = getattr(doc, 'metadata', {})
            logger.info(f'Doc {i}:')
            if len(content) > 200:
                logger.info(f'Content: {content[:200]}...')
            else:
                logger.info(f'Content: {content}')
            logger.info(f'Metadata: {metadata}')
            logger.info('-' * 50)

        # Search for specific DIN terms
        search_terms = ['0636-3', 'VDE', 'DIN VDE', 'Niederspannungssicherungen']
        search_vectorstore(vector_store, search_terms)

    except Exception as e:
        logger.error(f'Error checking vector store: {e}')
        raise


def search_vectorstore(vector_store, search_terms: List[str], k: int = 3) -> None:
    """Search the vector store for specific terms.

    Args:
        vector_store: Initialized ChromaVectorStore instance
        search_terms: List of terms to search for
        k: Number of results to return for each search

    """
    for term in search_terms:
        logger.info(f"Searching for '{term}'...")
        try:
            search_results = vector_store.similarity_search(term, k=k)

            msg = f"Found {len(search_results)} documents matching '{term}'"
            logger.info(msg)

            for i, doc in enumerate(search_results):
                # Document objects have page_content and metadata attributes
                content = getattr(doc, 'page_content', '')
                metadata = getattr(doc, 'metadata', {})
                logger.info(f'Result {i}:')
                if len(content) > 200:
                    logger.info(f'Content: {content[:200]}...')
                else:
                    logger.info(f'Content: {content}')
                logger.info(f'Metadata: {metadata}')
                logger.info('-' * 50)
        except Exception as e:
            logger.error(f"Error searching for term '{term}': {e}")


def main():
    """Run the vector store check."""
    try:
        check_vectorstore()
    except Exception as e:
        logger.error(f'Vector store check failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
