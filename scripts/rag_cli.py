#!/usr/bin/env python
"""RAG CLI - Command-line interface for the RAG system.

Usage:
    python -m scripts.rag_cli --vector-db <path> --collection-name <n>
    python -m scripts.rag_cli --vector-db <path> --collection-name <n> \
        --model <model>
    python -m scripts.rag_cli --vector-db <path> --collection-name <n> \
        --no-device-map
    python -m scripts.rag_cli --vector-db <path> --collection-name <n> \
        --use-llama --llama-model-path <path>
"""

import argparse
import logging
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after path modification
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from llm_rag.rag.pipeline import ConversationalRAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Command Line Interface for the RAG System")
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-base",
        help="HuggingFace model to use for generation",
    )
    parser.add_argument(
        "--vector-db",
        type=str,
        default="data/vector_db",
        help="Path to the vector database",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedding model to use",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="documents",
        help="Name of the vector database collection",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--use-llama",
        action="store_true",
        help="Use LLaMA model instead of HuggingFace model",
    )
    parser.add_argument(
        "--llama-model-path",
        type=str,
        default=None,
        help="Path to LLaMA model file (.gguf format)",
    )
    parser.add_argument(
        "--llama-n-ctx",
        type=int,
        default=2048,
        help="Context size for LLaMA model",
    )
    parser.add_argument(
        "--llama-n-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU for LLaMA model",
    )
    parser.add_argument(
        "--no-device-map",
        action="store_true",
        help="Disable device_map for model loading (no accelerate needed)",
    )
    return parser.parse_args()


def setup_llm(
    model_name="google/flan-t5-base",
    max_tokens=256,
    temperature=0.1,
    use_llama=False,
    llama_model_path=None,
    llama_n_ctx=2048,
    llama_n_gpu_layers=0,
    use_device_map=True,
):
    """Set up the language model for the RAG pipeline.

    Args:
    ----
        model_name: Name of the HuggingFace model to use
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        use_llama: Whether to use LlamaCpp
        llama_model_path: Path to the LlamaCpp model
        llama_n_ctx: Context window size for LlamaCpp
        llama_n_gpu_layers: Number of GPU layers for LlamaCpp
        use_device_map: Whether to use device_map when loading models

    Returns:
    -------
        Language model instance

    """
    if use_llama:
        if not llama_model_path:
            raise ValueError("llama_model_path must be provided when use_llama is True")

        try:
            return LlamaCpp(
                model_path=llama_model_path,
                n_ctx=llama_n_ctx,
                n_gpu_layers=llama_n_gpu_layers,
                verbose=False,
            )
        except ImportError as err:
            msg = "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            logger.error(msg)
            raise ImportError(msg) from err

    try:
        # Try to load the specified model
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with or without device_map based on user preference
        if use_device_map:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto",
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Create text generation pipeline
        text_generation = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
        )

        # Create LangChain LLM
        return HuggingFacePipeline(pipeline=text_generation)

    except Exception as e:
        # If loading fails, try a smaller model
        logger.warning(f"Failed to load {model_name}: {str(e)}. Falling back to google/flan-t5-small.")

        try:
            # Fallback to a smaller model
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

            # Load model with or without device_map based on user preference
            if use_device_map:
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", device_map="auto")
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

            text_generation = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

            return HuggingFacePipeline(pipeline=text_generation)

        except Exception as e2:
            logger.error(f"Error loading fallback model: {str(e2)}")
            logger.error("Could not load any model. Exiting.")
            sys.exit(1)


def load_vectorstore(vector_db_path, embedding_model, collection_name):
    """Load the vector store from disk."""
    logger.info(f"Loading vector store from: {vector_db_path}")

    try:
        # Check if vector store exists
        if not os.path.exists(vector_db_path):
            logger.error(f"Vector store not found at: {vector_db_path}")
            logger.info("Please load documents first using the load_documents.py script.")
            sys.exit(1)

        # Load embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Load vector store
        vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=embeddings,
            collection_name=collection_name,
        )

        return vectorstore

    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        sys.exit(1)


def main():
    """Run the RAG CLI.

    This function initializes the RAG pipeline and starts the interactive CLI.
    """
    args = parse_args()

    # Load vector store
    vectorstore = load_vectorstore(args.vector_db, args.embedding_model, args.collection_name)

    # Load language model
    llm = setup_llm(
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_llama=args.use_llama,
        llama_model_path=args.llama_model_path,
        llama_n_ctx=args.llama_n_ctx,
        llama_n_gpu_layers=args.llama_n_gpu_layers,
        use_device_map=not args.no_device_map,
    )

    # Create RAG pipeline
    rag_pipeline = ConversationalRAGPipeline(
        vectorstore=vectorstore,
        llm=llm,
        top_k=args.top_k,
    )

    # Welcome message
    print("\n" + "=" * 80)
    print("Welcome to the RAG CLI!")
    print("Ask questions about your documents or type 'exit' to quit.")
    print("=" * 80 + "\n")

    # Check if input is from a pipe or interactive terminal
    is_pipe = not sys.stdin.isatty()

    # For non-interactive mode, read all input at once
    if is_pipe:
        # Read all lines from stdin
        lines = sys.stdin.readlines()
        for line in lines:
            query = line.strip()
            if not query:
                continue

            print(f"\nYou: {query}")

            # Check for exit command
            if query.lower() in ["exit", "quit", "q"]:
                print("\nThank you for using the RAG CLI. Goodbye!")
                break

            try:
                # Process query
                result = rag_pipeline.query(query)

                # Print response
                print(f"\nAssistant: {result['response']}")

                # Print sources
                if "retrieved_documents" in result and result["retrieved_documents"]:
                    print("\nSources:")
                    for i, doc in enumerate(result["retrieved_documents"], 1):
                        source = doc.get("metadata", {}).get("source", "Unknown")
                        print(f"  {i}. {source}")
            except Exception as e:
                print(f"\nError: {str(e)}")
                logger.error(f"Error processing query: {str(e)}")

        return

    # Interactive mode
    while True:
        try:
            # Get user input
            query = input("\nYou: ")

            # Check for exit command
            if query.lower() in ["exit", "quit", "q"]:
                print("\nThank you for using the RAG CLI. Goodbye!")
                break

            # Process query
            result = rag_pipeline.query(query)

            # Print response
            print(f"\nAssistant: {result['response']}")

            # Print sources (optional)
            if "retrieved_documents" in result and result["retrieved_documents"]:
                print("\nSources:")
                for i, doc in enumerate(result["retrieved_documents"], 1):
                    source = doc.get("metadata", {}).get("source", "Unknown")
                    print(f"  {i}. {source}")

        except EOFError:
            print("\nInput stream ended. Exiting.")
            break

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break

        except Exception as e:
            print(f"\nError: {str(e)}")
            logger.error(f"Error processing query: {str(e)}")


if __name__ == "__main__":
    main()
