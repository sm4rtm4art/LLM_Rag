#!/usr/bin/env python
"""Demo script for the LLM RAG system using a local LLM.

This script demonstrates how to use the RAG system with a local LLM model.
It supports both interactive mode and single query mode.

Usage:
    python demo_llm_rag.py [--model-path MODEL_PATH] [--query QUERY]

Example:
-------
    # Interactive mode
    python demo_llm_rag.py --model-path models/llama-2-7b-chat.Q4_K_M.gguf

    # Single query mode
    python demo_llm_rag.py --model-path models/llama-2-7b-chat.Q4_K_M.gguf --query "What is RAG?"

"""

import argparse
import logging
import os
import sys

# Add the current directory to the path so we can import the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import RAG components
from llm_rag.main import CustomLlamaCpp  # noqa: E402
from llm_rag.rag.pipeline import BaseConversationalRAGPipeline, RAGPipeline  # noqa: E402
from llm_rag.vectorstore.chroma import ChromaVectorStore  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the demo script."""
    parser = argparse.ArgumentParser(description='Demo script for the LLM RAG system using a local LLM.')
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/llama-2-7b-chat.Q4_K_M.gguf',
        help='Path to the GGUF model file',
    )
    parser.add_argument(
        '--db-dir',
        type=str,
        default='chroma_db',
        help='Directory containing the ChromaDB vector store',
    )
    parser.add_argument(
        '--collection-name',
        type=str,
        default='documents',
        help='Name of the collection in ChromaDB',
    )
    parser.add_argument(
        '--n-gpu-layers',
        type=int,
        default=0,
        help='Number of layers to offload to GPU (0 for CPU only)',
    )
    parser.add_argument(
        '--n-ctx',
        type=int,
        default=2048,
        help='Context size for the model',
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode',
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Query to run (if not in interactive mode)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output',
    )
    return parser


def setup_llm(args: argparse.Namespace) -> CustomLlamaCpp:
    """Set up the language model."""
    if not os.path.exists(args.model_path):
        print(f'Error: Model file not found at {args.model_path}')
        print('Please download a GGUF model and place it in the models directory.')
        print('You can use the download_model.py script to download a model.')
        sys.exit(1)

    print(f'Loading model from {args.model_path}...')
    llm = CustomLlamaCpp(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        verbose=args.verbose,
    )
    return llm


def setup_vector_store(args: argparse.Namespace) -> ChromaVectorStore:
    """Set up the vector store."""
    if not os.path.exists(args.db_dir):
        print(f'Error: Vector store directory not found at {args.db_dir}')
        print('Please ingest some documents first.')
        sys.exit(1)

    print(f'Loading vector store from {args.db_dir}...')
    vector_store = ChromaVectorStore(
        persist_directory=args.db_dir,
        collection_name=args.collection_name,
    )
    return vector_store


def run_interactive_mode(rag_pipeline: BaseConversationalRAGPipeline) -> None:
    """Run the RAG pipeline in interactive mode."""
    print("\nEntering interactive mode. Type 'exit' to quit.")
    print('Type your query and press Enter.')

    while True:
        try:
            query = input('\nQuery: ')
            if query.lower() in ['exit', 'quit', 'q']:
                break

            if not query.strip():
                continue

            print('Thinking...')
            response = rag_pipeline.query(query)
            print(f'\nResponse: {response}')

        except KeyboardInterrupt:
            print('\nExiting...')
            break
        except Exception as e:
            print(f'Error: {e}')


def run_single_query(rag_pipeline: RAGPipeline, query: str) -> None:
    """Run a single query through the RAG pipeline."""
    print(f'Query: {query}')
    print('Thinking...')
    response = rag_pipeline.query(query)
    print(f'\nResponse: {response}')


def main() -> None:
    """Run the main demo script."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Validate arguments
    if not args.interactive and not args.query:
        parser.error('Either --interactive or --query must be specified')

    # Set up the language model
    llm = setup_llm(args)

    # Set up the vector store
    vector_store = setup_vector_store(args)

    # Create the RAG pipeline
    if args.interactive:
        print('Creating conversational RAG pipeline...')
        rag_pipeline = BaseConversationalRAGPipeline(
            vector_store=vector_store,
            llm_chain=llm,
        )
        run_interactive_mode(rag_pipeline)
    else:
        print('Creating RAG pipeline...')
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_chain=llm,
        )
        run_single_query(rag_pipeline, args.query)


if __name__ == '__main__':
    main()
