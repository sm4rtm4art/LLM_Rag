#!/usr/bin/env python3
"""Performance comparison between ChromaVectorStore and ShardedChromaVectorStore.

This script measures the performance of both vector store implementations
with different document collection sizes and configurations.

Usage:
    python -m demos.performance_test_sharded --doc-count 10000 --batch-size 1000

Arguments:
    --doc-count: Total number of documents to test with (default: 10000)
    --batch-size: Batch size for adding documents (default: 1000)
    --shard-capacity: Maximum number of documents per shard (default: 5000)
    --workers: Number of workers for concurrent search (default: 4)

"""

import argparse
import logging
import os
import random
import shutil
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from llm_rag.vectorstore import ShardedChromaVectorStore
from llm_rag.vectorstore.chroma import ChromaVectorStore, EmbeddingFunctionWrapper

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Performance test for ShardedChromaVectorStore")
    parser.add_argument("--doc-count", type=int, default=10000, help="Total number of documents to test with")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for adding documents")
    parser.add_argument("--shard-capacity", type=int, default=5000, help="Maximum number of documents per shard")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for concurrent search")
    return parser.parse_args()


def generate_sample_documents(num_docs: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Generate sample documents with metadata for testing."""
    documents = []
    metadatas = []
    categories = ["technical", "financial", "legal", "marketing", "research"]

    for i in range(num_docs):
        # Generate a simple document with some content
        doc_id = f"doc_{i:05d}"
        document = f"This is sample document {doc_id}. "
        document += "It contains some random text for demonstration purposes. "
        document += f"This document belongs to the {random.choice(categories)} "
        document += f"category. It has a unique identifier of {doc_id}."

        # Generate metadata for the document
        metadata = {
            "id": doc_id,
            "category": random.choice(categories),
            "length": len(document),
            "created_at": time.time(),
        }

        documents.append(document)
        metadatas.append(metadata)

    return documents, metadatas


def test_regular_chroma(
    total_docs: int, batch_size: int, persist_directory: str, embedding_function
) -> Dict[str, float]:
    """Test performance of regular ChromaVectorStore."""
    logger.info("Testing regular ChromaVectorStore")
    results = {
        "insert_time": 0.0,
        "search_time": 0.0,
        "docs_per_second": 0.0,
    }

    # Clean directory if it exists
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize the vector store
    vector_store = ChromaVectorStore(persist_directory=persist_directory, embedding_function=embedding_function)

    # Add documents in batches
    total_insert_time = 0.0
    docs_added = 0

    for batch_idx in range(0, total_docs, batch_size):
        batch_docs = min(batch_size, total_docs - docs_added)
        logger.info(f"Adding batch {batch_idx // batch_size + 1}, size: {batch_docs}")

        documents, metadatas = generate_sample_documents(batch_docs)

        # Time document insertion
        start_time = time.time()
        vector_store.add_documents(documents, metadatas)
        end_time = time.time()

        batch_time = end_time - start_time
        total_insert_time += batch_time
        docs_added += batch_docs

        logger.info(f"Batch inserted in {batch_time:.2f} seconds, {batch_docs / batch_time:.2f} docs/sec")

    results["insert_time"] = total_insert_time
    results["docs_per_second"] = total_docs / total_insert_time if total_insert_time > 0 else 0

    # Test search performance
    search_queries = [
        "technical document",
        "financial report",
        "marketing strategy",
        "legal compliance",
        "research findings",
    ]

    total_search_time = 0.0

    for query in search_queries:
        start_time = time.time()
        results_found = vector_store.search(query, n_results=10)
        end_time = time.time()

        query_time = end_time - start_time
        total_search_time += query_time

        logger.info(f"Query '{query}' completed in {query_time:.4f} seconds, found {len(results_found)} results")

    avg_search_time = total_search_time / len(search_queries)
    results["search_time"] = avg_search_time

    logger.info(
        f"Regular ChromaVectorStore results: "
        f"Insert: {results['insert_time']:.2f}s, "
        f"Search: {results['search_time']:.4f}s, "
        f"Throughput: {results['docs_per_second']:.2f} docs/sec"
    )

    return results


def test_sharded_chroma(
    total_docs: int, batch_size: int, shard_capacity: int, max_workers: int, persist_directory: str, embedding_function
) -> Dict[str, float]:
    """Test performance of ShardedChromaVectorStore."""
    logger.info("Testing ShardedChromaVectorStore")
    results = {"insert_time": 0.0, "search_time": 0.0, "docs_per_second": 0.0, "shard_count": 0}

    # Clean directory if it exists
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize the vector store
    vector_store = ShardedChromaVectorStore(
        shard_capacity=shard_capacity,
        base_persist_directory=persist_directory,
        max_workers=max_workers,
        embedding_function=embedding_function,
    )

    # Add documents in batches
    total_insert_time = 0.0
    docs_added = 0

    for batch_idx in range(0, total_docs, batch_size):
        batch_docs = min(batch_size, total_docs - docs_added)
        logger.info(f"Adding batch {batch_idx // batch_size + 1}, size: {batch_docs}")

        documents, metadatas = generate_sample_documents(batch_docs)

        # Time document insertion
        start_time = time.time()
        vector_store.add_documents(documents, metadatas)
        end_time = time.time()

        batch_time = end_time - start_time
        total_insert_time += batch_time
        docs_added += batch_docs

        logger.info(
            f"Batch inserted in {batch_time:.2f} seconds, "
            f"{batch_docs / batch_time:.2f} docs/sec, "
            f"shards: {len(vector_store.shards)}"
        )

    results["insert_time"] = total_insert_time
    results["docs_per_second"] = total_docs / total_insert_time if total_insert_time > 0 else 0
    results["shard_count"] = len(vector_store.shards)

    # Test search performance
    search_queries = [
        "technical document",
        "financial report",
        "marketing strategy",
        "legal compliance",
        "research findings",
    ]

    total_search_time = 0.0

    for query in search_queries:
        start_time = time.time()
        results_found = vector_store.search(query, n_results=10)
        end_time = time.time()

        query_time = end_time - start_time
        total_search_time += query_time

        logger.info(f"Query '{query}' completed in {query_time:.4f} seconds, found {len(results_found)} results")

    avg_search_time = total_search_time / len(search_queries)
    results["search_time"] = avg_search_time

    logger.info(
        f"ShardedChromaVectorStore results: "
        f"Insert: {results['insert_time']:.2f}s, "
        f"Search: {results['search_time']:.4f}s, "
        f"Throughput: {results['docs_per_second']:.2f} docs/sec, "
        f"Shards: {results['shard_count']}"
    )

    return results


def plot_results(regular_results, sharded_results, args):
    """Plot performance comparison results."""
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"ChromaVectorStore vs ShardedChromaVectorStore Performance\n"
        f"({args.doc_count} documents, shard capacity: {args.shard_capacity})",
        fontsize=16,
    )

    # Insert time comparison
    axs[0, 0].bar(
        ["Regular", "Sharded"],
        [regular_results["insert_time"], sharded_results["insert_time"]],
        color=["blue", "orange"],
    )
    axs[0, 0].set_title("Total Insert Time (seconds)")
    axs[0, 0].set_ylabel("Seconds")

    # Documents per second
    axs[0, 1].bar(
        ["Regular", "Sharded"],
        [regular_results["docs_per_second"], sharded_results["docs_per_second"]],
        color=["blue", "orange"],
    )
    axs[0, 1].set_title("Insert Throughput (docs/second)")
    axs[0, 1].set_ylabel("Documents per second")

    # Search time comparison
    axs[1, 0].bar(
        ["Regular", "Sharded"],
        [regular_results["search_time"], sharded_results["search_time"]],
        color=["blue", "orange"],
    )
    axs[1, 0].set_title("Average Search Time (seconds)")
    axs[1, 0].set_ylabel("Seconds")

    # Extra information
    axs[1, 1].axis("off")
    info_text = (
        f"Configuration:\n"
        f"Total documents: {args.doc_count}\n"
        f"Batch size: {args.batch_size}\n"
        f"Shard capacity: {args.shard_capacity}\n"
        f"Workers: {args.workers}\n\n"
        f"Regular Chroma:\n"
        f"  Insert time: {regular_results['insert_time']:.2f}s\n"
        f"  Search time: {regular_results['search_time']:.4f}s\n"
        f"  Throughput: {regular_results['docs_per_second']:.2f} docs/sec\n\n"
        f"Sharded Chroma:\n"
        f"  Insert time: {sharded_results['insert_time']:.2f}s\n"
        f"  Search time: {sharded_results['search_time']:.4f}s\n"
        f"  Throughput: {sharded_results['docs_per_second']:.2f} docs/sec\n"
        f"  Shards: {sharded_results['shard_count']}"
    )
    axs[1, 1].text(0.05, 0.95, info_text, verticalalignment="top", fontsize=10)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("chroma_performance_comparison.png", dpi=150)
    logger.info("Performance comparison chart saved to chroma_performance_comparison.png")

    # Print speedup factors
    insert_speedup = (
        regular_results["insert_time"] / sharded_results["insert_time"] if sharded_results["insert_time"] > 0 else 0
    )
    search_speedup = (
        regular_results["search_time"] / sharded_results["search_time"] if sharded_results["search_time"] > 0 else 0
    )
    throughput_speedup = (
        sharded_results["docs_per_second"] / regular_results["docs_per_second"]
        if regular_results["docs_per_second"] > 0
        else 0
    )

    logger.info("Performance summary:")
    logger.info(f"  Insert speedup: {insert_speedup:.2f}x")
    logger.info(f"  Search speedup: {search_speedup:.2f}x")
    logger.info(f"  Throughput speedup: {throughput_speedup:.2f}x")


def main():
    """Run performance tests and comparisons."""
    args = parse_args()

    logger.info(f"Running performance tests with {args.doc_count} documents")
    logger.info(f"Batch size: {args.batch_size}, Shard capacity: {args.shard_capacity}")

    # Initialize the embedding function (shared across tests)
    embedding_function = EmbeddingFunctionWrapper()

    # Test regular ChromaVectorStore
    regular_results = test_regular_chroma(
        total_docs=args.doc_count,
        batch_size=args.batch_size,
        persist_directory="data/performance_test/regular",
        embedding_function=embedding_function,
    )

    # Test ShardedChromaVectorStore
    sharded_results = test_sharded_chroma(
        total_docs=args.doc_count,
        batch_size=args.batch_size,
        shard_capacity=args.shard_capacity,
        max_workers=args.workers,
        persist_directory="data/performance_test/sharded",
        embedding_function=embedding_function,
    )

    # Plot and analyze results
    plot_results(regular_results, sharded_results, args)

    logger.info("Performance tests completed")


if __name__ == "__main__":
    main()
