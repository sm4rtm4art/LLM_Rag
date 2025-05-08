#!/usr/bin/env python
"""Simple test script for the RAG pipeline using a mock LLM.

This script demonstrates how to test the RAG pipeline without using a real LLM,
which makes testing much faster and simpler.
"""

import argparse
import logging
import os
import sys
import traceback
from typing import Any, List, Optional

# Add the parent directory to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

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
    parser = argparse.ArgumentParser(description='Test the RAG pipeline with a mock LLM.')
    parser.add_argument(
        '--query', type=str, default='What are RAG systems?', help='The query to test with the RAG pipeline.'
    )
    parser.add_argument(
        '--collection', type=str, default='documents', help='The name of the collection in the vector store.'
    )
    parser.add_argument(
        '--db-path', type=str, default='data/vectorstore', help='The path to the vector store database.'
    )
    parser.add_argument('--top-k', type=int, default=3, help='The number of documents to retrieve.')
    return parser.parse_args()


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

    # Execute the query
    query = args.query
    logger.info('Executing query: %s', query)
    try:
        # The query method returns a dictionary with the response
        # Note: The RAGPipeline has a hardcoded response for testing purposes,
        # but it still calls our MockLLM and stores the actual
        # response in the conversation history.
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
