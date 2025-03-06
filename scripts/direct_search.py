#!/usr/bin/env python
"""
Direct search script to test ChromaDB retrieval.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.llm_rag.vectorstore.chroma import ChromaVectorStore


def direct_search(
    query: str,
    db_path: str,
    collection_name: str,
    n_results: int = 10,
    where: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """Directly search the ChromaDB collection."""
    # Load the vector store
    logger.info(f"Loading vector store from {db_path}")
    vector_store = ChromaVectorStore(
        persist_directory=db_path,
        collection_name=collection_name,
    )
    
    # Search for documents
    logger.info(f"Searching for: {query}")
    results = vector_store.search(
        query=query,
        n_results=n_results,
        where=where,
    )
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Direct search in ChromaDB")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to search for",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="test_subset",
        help="Collection name",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="chroma_db",
        help="Path to ChromaDB database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of results to return",
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Filter by filename",
    )
    
    args = parser.parse_args()
    
    # Prepare where clause if filename is provided
    where = None
    if args.filename:
        where = {"filename": args.filename}
    
    # Search for documents
    results = direct_search(
        query=args.query,
        db_path=args.db_path,
        collection_name=args.collection,
        n_results=args.limit,
        where=where,
    )
    
    # Print results
    logger.info(f"Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"ID: {result['id']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Distance: {result['distance']}")
        print(f"Content preview: {result['document'][:200]}...")


if __name__ == "__main__":
    main() 