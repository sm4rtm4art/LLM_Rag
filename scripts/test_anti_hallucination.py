#!/usr/bin/env python
"""Test script for anti-hallucination improvements in the RAG system.

This script tests the anti-hallucination improvements by running queries
against the RAG system and comparing the results with and without the
anti-hallucination features.
"""

import argparse
import logging
import sys
from typing import Any, Dict

from src.llm_rag.models.factory import ModelBackend, ModelFactory
from src.llm_rag.rag.anti_hallucination import (
    extract_key_entities,
)
from src.llm_rag.rag.pipeline import RAGPipeline
from src.llm_rag.vectorstore.chroma import ChromaVectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the script."""
    parser = argparse.ArgumentParser(description='Test anti-hallucination improvements in the RAG system.')
    parser.add_argument(
        '--model_name',
        type=str,
        default='microsoft/phi-2',
        help='HuggingFace model to use',
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help='ChromaDB collection name',
    )
    parser.add_argument(
        '--db_path',
        type=str,
        default='chroma_db',
        help='Path to ChromaDB database',
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Query to ask the RAG system',
    )
    parser.add_argument(
        '--test_queries',
        action='store_true',
        help='Run a set of test queries',
    )
    return parser


def run_rag_query(
    query: str,
    model_name: str,
    db_path: str,
    collection_name: str,
) -> Dict[str, Any]:
    """Run a RAG query against the system."""
    # Load the LLM model
    logger.info(f'Loading model: {model_name}')
    llm = ModelFactory.create_model(
        model_path_or_name=model_name,
        backend=ModelBackend.HUGGINGFACE,
        device='cpu',
        max_tokens=512,
        temperature=0.7,
    )

    # Load the vector store
    logger.info(f'Loading vector store from {db_path}')
    vector_store = ChromaVectorStore(db_path, collection_name)

    # Create the RAG pipeline
    logger.info('Creating RAG pipeline')
    pipeline = RAGPipeline(
        vectorstore=vector_store,
        llm=llm,
        top_k=3,
    )

    # Run the query
    logger.info(f'Running query: {query}')
    response = pipeline.query(query)

    # Log detailed information about the response for debugging
    logger.info(f'Response type: {type(response)}')
    logger.info(f'Response content: {response}')

    if 'response' in response:
        logger.info(f"Response['response'] type: {type(response['response'])}")
        logger.info(f"Response['response'] content: {response['response']}")

    # Log retrieved documents for debugging
    if 'documents' in response:
        logger.info(f'Number of retrieved documents: {len(response["documents"])}')
        for i, doc in enumerate(response['documents']):
            logger.info(f'Document {i + 1}:')
            logger.info(f'  Content (first 100 chars): {doc.page_content[:100]}...')
            logger.info(f'  Metadata: {doc.metadata}')

    return response


def analyze_response(response: str, context: str) -> None:
    """Analyze a response for potential hallucinations."""
    # Extract entities
    response_entities = extract_key_entities(response)
    context_entities = extract_key_entities(context)

    # Find missing entities
    missing_entities = [entity for entity in response_entities if entity not in context_entities]

    # Calculate coverage ratio
    if response_entities:
        coverage_ratio = 1.0 - (len(missing_entities) / len(response_entities))
    else:
        coverage_ratio = 1.0

    # Print analysis
    print('\nEntity Analysis:')
    print(f'  Response entities: {len(response_entities)}')
    print(f'  Context entities: {len(context_entities)}')
    print(f'  Missing entities: {len(missing_entities)}')
    print(f'  Coverage ratio: {coverage_ratio:.2f}')

    if missing_entities:
        print('\nPotentially hallucinated entities:')
        for entity in missing_entities[:10]:
            print(f'  - {entity}')
        if len(missing_entities) > 10:
            print(f'  ... and {len(missing_entities) - 10} more')


def run_test_queries(model_name: str, db_path: str, collection_name: str) -> None:
    """Run a set of test queries to evaluate anti-hallucination performance."""
    test_queries = [
        'What is the purpose of DIN_SPEC_31009?',
        'What are the safety requirements for toys according to EN 71-1?',
        'How does DIN_SPEC_31009 handle flammability requirements?',
        'What is the relationship between DIN_SPEC_31009 and polyurethane foam insulation?',
        'What does DIN_SPEC_31009 say about protective devices?',
    ]

    for query in test_queries:
        print('\n' + '=' * 80)
        print(f'QUERY: {query}')
        print('=' * 80)

        result = run_rag_query(
            query=query,
            model_name=model_name,
            db_path=db_path,
            collection_name=collection_name,
        )

        response = result['response']
        confidence = result.get('confidence', 0.0)
        documents = result.get('documents', [])

        # Format context for analysis
        context = '\n\n'.join([doc.get('content', '') for doc in documents])

        print('\nRAG System Response:')
        print('-------------------')
        print(response)
        print('-------------------')

        # Display confidence score
        confidence_level = 'High' if confidence >= 0.8 else 'Medium' if confidence >= 0.5 else 'Low'
        print(f'\nRetrieval Confidence: {confidence:.2f} ({confidence_level})')

        # Analyze the response
        analyze_response(response, context)


def main() -> None:
    """Run the main script logic."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    if args.test_queries:
        run_test_queries(
            model_name=args.model_name,
            db_path=args.db_path,
            collection_name=args.collection,
        )
        return

    if not args.query:
        query = input('Enter your query: ')
    else:
        query = args.query

    try:
        result = run_rag_query(
            query=query,
            model_name=args.model_name,
            db_path=args.db_path,
            collection_name=args.collection,
        )

        response = result['response']
        confidence = result.get('confidence', 0.0)
        documents = result.get('documents', [])

        # Format context for analysis
        context = '\n\n'.join([doc.get('content', '') for doc in documents])

        print('\nRAG System Response:')
        print('-------------------')
        print(response)
        print('-------------------')

        # Display confidence score
        confidence_level = 'High' if confidence >= 0.8 else 'Medium' if confidence >= 0.5 else 'Low'
        print(f'\nRetrieval Confidence: {confidence:.2f} ({confidence_level})')

        # Display document sources
        if documents:
            print('\nRetrieved from:')
            for i, doc in enumerate(documents):
                metadata = doc.get('metadata', {})
                source = metadata.get('source', 'Unknown')
                filename = metadata.get('filename', '')
                page = metadata.get('page', '')

                source_info = []
                if source and source != 'Unknown':
                    source_info.append(f'Source: {source}')
                if filename:
                    source_info.append(f'File: {filename}')
                if page:
                    source_info.append(f'Page: {page}')

                source_display = ', '.join(source_info) if source_info else 'Unknown source'
                print(f'  {i + 1}. {source_display}')

        # Analyze the response
        analyze_response(response, context)

    except Exception as e:
        logger.error(f'Error running RAG query: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
