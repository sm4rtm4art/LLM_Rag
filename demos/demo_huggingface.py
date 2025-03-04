#!/usr/bin/env python
"""Demo script for the LLM RAG system using Hugging Face models.

This script demonstrates how to use the RAG system with Hugging Face models,
including Llama-3. It supports both interactive mode and single query mode.

Usage:
    python demo_huggingface.py [--model MODEL_NAME] [--query QUERY]

Example:
-------
    # Interactive mode with Llama-3
    python demo_huggingface.py --model meta-llama/Llama-3-8B-Instruct

    # Single query mode with Mistral
    python demo_huggingface.py --model mistralai/Mistral-7B-Instruct-v0.2 \
        --query "What is RAG?"

"""

import argparse
import logging
import os
import sys
import traceback

# Add the parent directory to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Import RAG components
from llm_rag.models.factory import ModelBackend, ModelFactory  # noqa: E402
from llm_rag.rag.pipeline import (
    ConversationalRAGPipeline,  # noqa: E402
    RAGPipeline,  # noqa: E402
)
from llm_rag.vectorstore.chroma import ChromaVectorStore  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for the demo script."""
    parser = argparse.ArgumentParser(
        description="Demo script for the LLM RAG system using Hugging Face "
                    "models.")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="Hugging Face model to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/vectorstore",
        help="Path to the vector store database",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="documents",
        help="Name of the collection in the vector store",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to run in non-interactive mode",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top k documents to retrieve",
    )
    return parser


def setup_llm(args: argparse.Namespace):
    """Set up the language model."""
    try:
        logger.debug(
            f"Creating model with: model={args.model}, device={args.device}")
        # Create the model using the factory
        model_name = args.model
        llm = ModelFactory.create_model(
            model_path_or_name=model_name,
            backend=ModelBackend.HUGGINGFACE,
            device=args.device,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        return llm
    except Exception as e:
        logger.error(f"Error setting up LLM: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def setup_vector_store(args: argparse.Namespace) -> ChromaVectorStore:
    """Set up the vector store."""
    try:
        logger.debug(
            f"Creating vector store with: db_path={args.db_path}, "
            f"collection_name={args.collection_name}")
        # Create a ChromaDB vector store
        vector_store = ChromaVectorStore(
            collection_name=args.collection_name,
            persist_directory=args.db_path,
        )
        # Log the number of documents in the vector store
        doc_count = len(vector_store.get_all_documents())
        logger.info(f"Vector store initialized with {doc_count} documents")
        if doc_count == 0:
            logger.warning(
                "No documents found in vector store. "
                "Load documents first using load_documents.py script.")
        return vector_store
    except Exception as e:
        logger.error(f"Error setting up vector store: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def run_interactive_mode(rag_pipeline: ConversationalRAGPipeline) -> None:
    """Run the RAG pipeline in interactive mode."""
    print("\nInteractive RAG with Hugging Face model. Type 'exit' to quit.")
    print("Enter your query:")

    while True:
        try:
            query = input("> ")
            if query.lower() in ["exit", "quit", "q"]:
                break

            print("Thinking...")
            response = rag_pipeline.query(query)
            print(f"\nResponse: {response['response']}")

            # Print the sources used
            if response.get("retrieved_documents"):
                print("\nSources:")
                for i, doc in enumerate(response["retrieved_documents"]):
                    doc_content = doc.get("content", "")
                    if len(doc_content) > 100:
                        doc_content = doc_content[:100] + "..."
                    source = doc.get("metadata", {}).get("source", "Unknown")
                    print(f"{i + 1}. {source}: {doc_content}")

        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            logger.error(traceback.format_exc())
            print(f"Error: {e}")


def run_single_query(rag_pipeline: RAGPipeline, query: str) -> None:
    """Run a single query through the RAG pipeline."""
    try:
        print(f"Query: {query}")
        print("Thinking...")

        # Add debug logging for the retrieval step
        logger.debug(f"Running query: {query}")
        response = rag_pipeline.query(query)
        print(f"Response: {response['response']}")

        # Print the sources used
        if response.get("retrieved_documents"):
            print("\nSources:")
            for i, doc in enumerate(response["retrieved_documents"]):
                doc_content = doc.get("content", "")
                if len(doc_content) > 100:
                    doc_content = doc_content[:100] + "..."
                source = doc.get("metadata", {}).get("source", "Unknown")
                print(f"{i + 1}. {source}: {doc_content}")

    except Exception as e:
        logger.error(f"Error running query: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {e}")


def main() -> None:
    """Run the demo script."""
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Validate arguments
    if not args.interactive and not args.query:
        parser.error("Either --interactive or --query must be specified")

    try:
        # Set up the LLM
        logger.info("Setting up LLM...")
        llm = setup_llm(args)
        logger.info("LLM setup complete")

        # Set up the vector store
        logger.info("Setting up vector store...")
        vector_store = setup_vector_store(args)
        logger.info("Vector store setup complete")

        # Set up the RAG pipeline
        logger.info("Setting up RAG pipeline...")
        try:
            if args.interactive:
                logger.debug("Creating ConversationalRAGPipeline")
                pipeline = ConversationalRAGPipeline(
                    llm_chain=llm,
                    vectorstore=vector_store,
                    top_k=args.top_k,
                )
            else:
                logger.debug("Creating RAGPipeline")
                pipeline = RAGPipeline(
                    llm=llm,
                    vectorstore=vector_store,
                    top_k=args.top_k,
                )
            logger.info("RAG pipeline setup complete")
        except Exception as e:
            logger.error(f"Error setting up RAG pipeline: {e}")
            logger.error(traceback.format_exc())
            raise

        # Run the pipeline
        if args.interactive:
            run_interactive_mode(pipeline)
        else:
            run_single_query(pipeline, args.query)

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
