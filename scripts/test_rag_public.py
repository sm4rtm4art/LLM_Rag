#!/usr/bin/env python
"""Test script for the RAG pipeline using public test data.

This script demonstrates how to test the RAG pipeline with public test data
that can be safely shared on GitHub. It uses a mock LLM for testing.
"""

import argparse
import json
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

# Add the parent directory to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import required components
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from llm_rag.document_processing.loaders import TextFileLoader
from llm_rag.rag.pipeline import RAGPipeline
from llm_rag.vectorstore.chroma import ChromaVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class MockLLM(LLM):
    """A mock LLM for testing the RAG pipeline."""

    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return 'mock_llm'

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the mock LLM with the given prompt."""
        logger.info('Mock LLM received prompt: %s', prompt[:100] + '...')

        # Extract the question from the prompt
        # This assumes the prompt follows the default template
        if 'Question:' in prompt and 'Answer:' in prompt:
            question = prompt.split('Question:')[1].split('Answer:')[0].strip()
        else:
            question = prompt

        # Return a mock response
        return f"This is a mock response to the question: '{question}'. The RAG pipeline is working correctly!"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Test the RAG pipeline with public test data.')
    parser.add_argument('--query', type=str, default='What is RAG?', help='The query to test with the RAG pipeline.')
    parser.add_argument(
        '--collection', type=str, default='public_test', help='The name of the collection in the vector store.'
    )
    parser.add_argument(
        '--db-path', type=str, default='data/vectorstore', help='The path to the vector store database.'
    )
    parser.add_argument('--top-k', type=int, default=3, help='The number of documents to retrieve.')
    parser.add_argument(
        '--test-queries-file',
        type=str,
        default='tests/test_data/test_queries.json',
        help='Path to the test queries file.',
    )
    parser.add_argument(
        '--use-test-query',
        type=int,
        default=None,
        help='Index of the test query to use from the test queries file (0-based).',
    )
    parser.add_argument(
        '--load-documents',
        action='store_true',
        help='Load the public test documents into the vector store before testing.',
    )
    return parser.parse_args()


def load_public_test_documents(vector_store):
    """Load public test documents into the vector store."""
    test_data_dir = 'tests/test_data'
    test_files = ['sample.txt', 'llama3_info.txt', 'rag_systems.txt']

    for file_name in test_files:
        file_path = os.path.join(test_data_dir, file_name)
        logger.info('Loading documents from %s', file_path)

        try:
            # Load documents using TextFileLoader
            loader = TextFileLoader(file_path)
            documents = loader.load()

            if documents:
                # Extract content and metadata from documents
                doc_contents = []
                doc_metadatas = []

                for doc in documents:
                    doc_contents.append(doc.page_content)
                    doc_metadatas.append(doc.metadata)

                # Add documents to vector store
                vector_store.add_documents(documents=doc_contents, metadatas=doc_metadatas)

                logger.info(f'Added {len(documents)} documents from {file_path}')
            else:
                logger.warning('No documents found in %s', file_path)
        except Exception as e:
            logger.error('Error loading documents from %s: %s', file_path, e)
            traceback.print_exc()


def load_test_queries(file_path: str) -> List[Dict[str, str]]:
    """Load test queries from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            queries = json.load(f)
        logger.info('Loaded %d test queries from %s', len(queries), file_path)
        return queries
    except Exception as e:
        logger.error('Error loading test queries from %s: %s', file_path, e)
        return []


def main():
    """Run the test script."""
    # Parse command-line arguments
    args = parse_args()

    # Set up the vector store
    logger.info('Setting up vector store...')
    vector_store = ChromaVectorStore(
        collection_name=args.collection,
        persist_directory=args.db_path,
    )
    logger.info('Vector store setup complete')

    # Load public test documents if requested
    if args.load_documents:
        logger.info('Loading public test documents...')
        load_public_test_documents(vector_store)
        logger.info('Public test documents loaded')

    # Set up the mock LLM
    logger.info('Setting up mock LLM...')
    llm = MockLLM()
    logger.info('Mock LLM setup complete')

    # Set up the RAG pipeline
    logger.info('Setting up RAG pipeline...')
    pipeline = RAGPipeline(
        vectorstore=vector_store,
        llm=llm,
        top_k=args.top_k,
    )
    logger.info('RAG pipeline setup complete')

    # Get the query
    query = args.query

    # If a test query index is specified, load the test queries and use the specified one
    if args.use_test_query is not None:
        test_queries = load_test_queries(args.test_queries_file)
        if test_queries and 0 <= args.use_test_query < len(test_queries):
            query = test_queries[args.use_test_query]['query']
            logger.info('Using test query %d: %s', args.use_test_query, query)
        else:
            logger.warning('Invalid test query index: %d. Using default query.', args.use_test_query)

    # Execute the query
    logger.info('Executing query: %s', query)
    try:
        result = pipeline.query(query)
        logger.info('Query executed successfully')
        logger.info('Response: %s', result)
        logger.info('Conversation history: %s', pipeline.conversation_history)
        logger.info('Test completed successfully')
    except Exception as e:
        logger.error('Error executing query: %s', e)
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
