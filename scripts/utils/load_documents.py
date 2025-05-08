#!/usr/bin/env python
"""Load documents into the RAG system.

This script loads documents from a specified directory into the RAG system's
vector store database, making them available for retrieval during queries.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List

# Add the project root to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Import our custom document loaders
from llm_rag.document_processing.chunking import RecursiveTextChunker  # noqa: E402
from llm_rag.document_processing.loaders import DirectoryLoader  # noqa: E402
from llm_rag.vectorstore.chroma import ChromaVectorStore  # noqa: E402


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the script."""
    parser = argparse.ArgumentParser(description="Load documents into the RAG system's vector store.")
    parser.add_argument(
        '--dir',
        type=str,
        default='data/documents/test_subset',
        help='Directory containing documents to load',
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='chroma_db',
        help='Path to the ChromaDB database',
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default='documents',
        help='Name of the ChromaDB collection',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Size of document chunks for splitting',
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Overlap between document chunks',
    )
    parser.add_argument(
        '--glob',
        type=str,
        default='**/*.txt',
        help='Glob pattern for matching files',
    )
    parser.add_argument(
        '--use-enhanced-pdf',
        action='store_true',
        help='Use enhanced PDF loader with table and image extraction',
    )
    parser.add_argument(
        '--din-standard-mode',
        action='store_true',
        help='Use DIN standard loader for DIN standard documents',
    )
    return parser


def load_documents(
    dir_path: str, glob_pattern: str, use_enhanced_pdf: bool = False, din_standard_mode: bool = False
) -> List:
    """Load documents from a directory.

    Args:
    ----
        dir_path: Path to the directory containing documents.
        glob_pattern: Pattern for matching files.
        use_enhanced_pdf: Whether to use enhanced PDF loader.
        din_standard_mode: Whether to use DIN standard loader.

    Returns:
    -------
        List of loaded documents.

    """
    logger.info(f'Loading documents from {dir_path} with pattern {glob_pattern}')

    # Create directory loader with our custom implementation
    loader = DirectoryLoader(
        directory_path=dir_path,
        recursive=True,
        glob_pattern=glob_pattern,
        use_enhanced_pdf=use_enhanced_pdf,
        din_standard_mode=din_standard_mode,
    )

    try:
        documents = loader.load()
        logger.info(f'Loaded {len(documents)} documents')
        return documents
    except Exception as e:
        logger.error(f'Error loading documents: {str(e)}')
        return []


def split_documents(documents: List, chunk_size: int, chunk_overlap: int) -> List:
    """Split documents into chunks.

    Args:
    ----
        documents: List of documents to split.
        chunk_size: Size of chunks.
        chunk_overlap: Overlap between chunks.

    Returns:
    -------
        List of document chunks.

    """
    if not documents:
        logger.warning('No documents to split')
        return []

    logger.info(f'Splitting {len(documents)} documents into chunks')

    # Use RecursiveTextChunker
    logger.info('Using RecursiveTextChunker')
    chunker = RecursiveTextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split documents
    document_chunks = chunker.split_documents(documents)
    logger.info(f'Created {len(document_chunks)} chunks')

    return document_chunks


def prepare_documents_for_vectorstore(
    document_chunks: List[Dict[str, Any]],
) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Prepare document chunks for adding to the vector store.

    Args:
    ----
        document_chunks: List of document chunks with 'content' and 'metadata' keys

    Returns:
    -------
        Tuple of (texts, metadatas, ids) ready for the vector store

    """
    texts = []
    metadatas = []
    ids = []

    for i, doc in enumerate(document_chunks):
        # Extract content and metadata
        if isinstance(doc, dict) and 'content' in doc and 'metadata' in doc:
            texts.append(doc['content'])
            metadatas.append(doc['metadata'])
        else:
            # Handle unexpected format
            logger.warning(f'Unexpected document format: {type(doc)}')
            continue

        # Generate ID
        doc_id = f'doc_{i}'
        ids.append(doc_id)

    logger.info(f'Prepared {len(texts)} documents for vector store')
    return texts, metadatas, ids


def main() -> None:
    """Execute the document loading script."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Check if the document directory exists
    if not os.path.exists(args.dir):
        logger.error(f'Document directory not found: {args.dir}')
        sys.exit(1)

    # Create the output directory if it doesn't exist
    os.makedirs(args.db_path, exist_ok=True)

    # Load documents
    documents = load_documents(
        args.dir, args.glob, use_enhanced_pdf=args.use_enhanced_pdf, din_standard_mode=args.din_standard_mode
    )

    if not documents:
        logger.error('No documents were loaded. Check your input directory and glob pattern.')
        sys.exit(1)

    # Split documents into chunks
    document_chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)

    if not document_chunks:
        logger.error('No document chunks were created. Check your input documents.')
        sys.exit(1)

    # Prepare documents for vector store
    texts, metadatas, ids = prepare_documents_for_vectorstore(document_chunks)

    if not texts:
        logger.error('Failed to prepare documents for vector store.')
        sys.exit(1)

    # Create vector store
    logger.info(f'Creating vector store at {args.db_path} with collection {args.collection_name}')

    # Use standard vector store
    logger.info('Using ChromaVectorStore')
    vector_store = ChromaVectorStore(
        collection_name=args.collection_name,
        persist_directory=args.db_path,
    )

    # Add documents to vector store
    logger.info('Adding documents to vector store...')

    # Use the collection.add method directly with prepared data
    vector_store.collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info(f'Successfully added {len(texts)} document chunks to the vector store')
    logger.info(f'Vector store path: {args.db_path}')
    logger.info(f'Collection name: {args.collection_name}')
    logger.info('You can now use the RAG system with the loaded documents')


if __name__ == '__main__':
    main()
