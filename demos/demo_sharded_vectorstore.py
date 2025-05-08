r"""Demo script for ShardedChromaVectorStore.

This script demonstrates the functionality of the ShardedChromaVectorStore,
including automatic shard creation, document addition, and search capabilities.

Usage:
    python -m demos.demo_sharded_vectorstore --shard-capacity 1000 \
        --docs-per-batch 300 --num-batches 5

Arguments:
    --shard-capacity: Maximum number of documents per shard (default: 1000)
    --docs-per-batch: Number of documents to add in each batch (default: 300)
    --num-batches: Number of batches to add (default: 5)
    --persist-directory: Base directory for storing shards
        (default: data/sharded_db)
    --max-workers: Number of workers for concurrent search (default: 4)

"""

import argparse
import logging
import os
import random
import shutil
import time
from typing import Any, Dict, List, Tuple

from llm_rag.vectorstore import ShardedChromaVectorStore
from llm_rag.vectorstore.chroma import EmbeddingFunctionWrapper

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Demo for ShardedChromaVectorStore')
    parser.add_argument('--shard-capacity', type=int, default=1000, help='Maximum number of documents per shard')
    parser.add_argument('--docs-per-batch', type=int, default=300, help='Number of documents to add in each batch')
    parser.add_argument('--num-batches', type=int, default=5, help='Number of batches to add')
    parser.add_argument(
        '--persist-directory', type=str, default='data/sharded_db', help='Base directory for storing shards'
    )
    parser.add_argument('--max-workers', type=int, default=4, help='Number of workers for concurrent search')
    parser.add_argument('--clean', action='store_true', help='Clean the persist directory before starting')
    return parser.parse_args()


def generate_sample_documents(num_docs: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Generate sample documents with metadata for testing."""
    documents = []
    metadatas = []
    categories = ['technical', 'financial', 'legal', 'marketing', 'research']

    for i in range(num_docs):
        # Generate a simple document with some content
        doc_id = f'doc_{i:05d}'
        document = f'This is sample document {doc_id}. '
        document += 'It contains some random text for demonstration purposes. '
        document += f'This document belongs to the {random.choice(categories)} '
        document += f'category. It has a unique identifier of {doc_id}.'

        # Generate metadata for the document
        metadata = {
            'id': doc_id,
            'category': random.choice(categories),
            'length': len(document),
            'created_at': time.time(),
        }

        documents.append(document)
        metadatas.append(metadata)

    return documents, metadatas


def count_shard_directories(persist_directory: str) -> int:
    """Count the number of shard directories created."""
    if not os.path.exists(persist_directory):
        return 0

    # Shards are named like "shard_0", "shard_1", etc.
    shard_dirs = [
        d
        for d in os.listdir(persist_directory)
        if os.path.isdir(os.path.join(persist_directory, d)) and d.startswith('shard_')
    ]

    return len(shard_dirs)


def main():
    """Run the ShardedChromaVectorStore demo."""
    args = parse_args()

    # Clean persist directory if requested
    if args.clean and os.path.exists(args.persist_directory):
        logger.info(f'Cleaning persist directory: {args.persist_directory}')
        shutil.rmtree(args.persist_directory)

    # Create persist directory if it doesn't exist
    os.makedirs(args.persist_directory, exist_ok=True)

    logger.info('Initializing embedding function...')
    embedding_function = EmbeddingFunctionWrapper()

    logger.info(
        f'Creating ShardedChromaVectorStore with '
        f'capacity={args.shard_capacity}, workers={args.max_workers}, '
        f'directory={args.persist_directory}'
    )

    # Initialize the ShardedChromaVectorStore
    vector_store = ShardedChromaVectorStore(
        shard_capacity=args.shard_capacity,
        base_persist_directory=args.persist_directory,
        max_workers=args.max_workers,
        embedding_function=embedding_function,
    )

    # Process documents in batches
    total_docs = 0
    for batch in range(args.num_batches):
        logger.info(f'Processing batch {batch + 1}/{args.num_batches}')

        # Generate sample documents
        num_docs = args.docs_per_batch
        logger.info(f'Generating {num_docs} sample documents')
        documents, metadatas = generate_sample_documents(num_docs)

        # Count shards before adding documents
        shard_count_before = len(vector_store.shards)
        logger.info(f'Current shard count: {shard_count_before}')

        # Add documents to the vector store
        doc_count = len(documents)
        logger.info(f'Adding {doc_count} documents to the vector store')
        start_time = time.time()
        vector_store.add_documents(documents, metadatas)
        elapsed_time = time.time() - start_time

        # Count shards after adding documents
        shard_count_after = len(vector_store.shards)
        logger.info(f'New shard count: {shard_count_after}')
        if shard_count_after > shard_count_before:
            new_shards = shard_count_after - shard_count_before
            logger.info(f'Created {new_shards} new shard(s)')

        # Update total document count
        total_docs += len(documents)

        # Check document count from vector store
        store_count = vector_store.count()
        logger.info(f'Total documents in store: {store_count}')

        # Verify directories on disk
        shard_dirs = count_shard_directories(args.persist_directory)
        logger.info(f'Number of shard directories on disk: {shard_dirs}')

        elapsed_fmt = f'{elapsed_time:.2f}'
        logger.info(f'Batch {batch + 1} completed in {elapsed_fmt} seconds')
        logger.info('-' * 40)

    # Perform searches to test functionality
    logger.info('Performing sample searches')
    search_queries = ['technical document', 'financial report', 'marketing strategy']

    for query in search_queries:
        logger.info(f"Searching for: '{query}'")
        start_time = time.time()
        results = vector_store.search(query, n_results=5)
        elapsed_time = time.time() - start_time

        result_count = len(results)
        elapsed_fmt = f'{elapsed_time:.2f}'
        logger.info(f'Found {result_count} results in {elapsed_fmt} seconds')
        for i, result in enumerate(results):
            # Handle different result formats
            if 'score' in result:
                score = f'{result["score"]:.4f}'
                doc_id = result['metadata'].get('id', 'unknown')
                category = result['metadata'].get('category', 'unknown')
            else:
                # Format for ChromaDB results
                score = f'{result.get("distance", 0.0):.4f}'
                doc_id = result.get('metadata', {}).get('id', 'unknown')
                category = result.get('metadata', {}).get('category', 'unknown')

            logger.info(f'Result {i + 1}: Score={score}, ID={doc_id}, Category={category}')
        logger.info('-' * 40)

    # Display final statistics
    logger.info('Final Statistics:')
    logger.info(f'Total documents added: {total_docs}')
    store_count = vector_store.count()
    logger.info(f'Document count from vector_store.count(): {store_count}')
    shard_count = len(vector_store.shards)
    logger.info(f'Number of shards in memory: {shard_count}')
    shard_dirs = count_shard_directories(args.persist_directory)
    logger.info(f'Number of shard directories on disk: {shard_dirs}')

    # Create a retriever from the vector store
    logger.info('Creating retriever from vector store')
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})

    # Test the retriever
    logger.info("Testing retriever with query: 'technical document'")
    docs = retriever.get_relevant_documents('technical document')
    doc_count = len(docs)
    logger.info(f'Retrieved {doc_count} documents using retriever')

    logger.info('Demo completed successfully')


if __name__ == '__main__':
    main()
