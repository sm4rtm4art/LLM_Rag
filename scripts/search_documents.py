#!/usr/bin/env python
"""Script to search for specific documents in the Chroma database."""

import chromadb
import argparse
from sentence_transformers import SentenceTransformer


def main():
    """Main function to search for documents."""
    parser = argparse.ArgumentParser(description="Search for documents in Chroma DB")
    parser.add_argument(
        "--collection",
        type=str,
        default="test_subset",
        help="Collection name to search",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to search for",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of documents to return",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="./chroma_db",
        help="Path to Chroma DB",
    )
    args = parser.parse_args()

    # Connect to the database
    client = chromadb.PersistentClient(args.db_path)
    
    # Get the collection
    collection = client.get_collection(
        args.collection,
        embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    
    # Search for documents
    results = collection.query(
        query_texts=[args.query],
        n_results=args.limit,
    )
    
    # Print results
    print(f"Search results for query: '{args.query}'")
    print(f"Number of results: {len(results['documents'][0])}")
    
    for i, (doc_id, metadata, document) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["documents"][0])
    ):
        print(f"\nResult {i+1} (ID: {doc_id}):")
        print(f"Metadata: {metadata}")
        print(f"Content preview: {document[:200]}...")
        print(f"Distance: {results['distances'][0][i]}")


if __name__ == "__main__":
    main() 