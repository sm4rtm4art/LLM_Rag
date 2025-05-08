#!/usr/bin/env python
"""Script to check document metadata in the Chroma database."""

import argparse

import chromadb


def main():
    """Check document metadata in the Chroma DB."""
    parser = argparse.ArgumentParser(description='Check document metadata in Chroma DB')
    parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help='Collection name to check',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Number of documents to check',
    )
    parser.add_argument(
        '--db_path',
        type=str,
        default='./chroma_db',
        help='Path to Chroma DB',
    )
    args = parser.parse_args()

    # Connect to the database
    client = chromadb.PersistentClient(args.db_path)

    # Get the collection
    collection = client.get_collection(args.collection)

    # Get document count
    doc_count = collection.count()
    print(f"Number of documents in collection '{args.collection}': {doc_count}")

    # Get sample documents
    sample = collection.get(limit=args.limit)

    # Print document metadata
    print('\nSample document metadata:')
    for i, (doc_id, metadata, document) in enumerate(
        zip(sample['ids'], sample['metadatas'], sample['documents'], strict=False)
    ):
        print(f'\nDocument {i} (ID: {doc_id}):')
        print(f'Metadata: {metadata}')
        print(f'Content preview: {document[:100]}...')


if __name__ == '__main__':
    main()
