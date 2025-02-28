"""Main module for the llm-rag package.

This module provides the main entry point for the llm-rag application.
It demonstrates how to set up and use a simple RAG pipeline.
"""
import argparse
import os
import sys

import chromadb
from langchain_community.llms import LlamaCpp

from llm_rag.document_processing.chunking import RecursiveTextChunker
from llm_rag.document_processing.loaders import DirectoryLoader
from llm_rag.models.embeddings import EmbeddingModel
from llm_rag.rag.pipeline import RAGPipeline
from llm_rag.vectorstore.chroma import ChromaVectorStore


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser.

    Returns
    -------
        argparse.ArgumentParser: The configured argument parser.

    """
    parser = argparse.ArgumentParser(description="LLM RAG System - Retrieval-Augmented Generation")
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to process (if not provided, interactive mode is used)",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default="./chroma_db",
        help="Directory to store the vector database",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/llama-2-7b-chat.gguf",
        help="Path to the Llama model file",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model to use for embeddings",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=1,
        help="Number of layers to offload to GPU",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        help="Context window size",
    )
    return parser


def ingest_documents(
    data_dir: str,
    db_dir: str,
    embedding_model_name: str,
) -> ChromaVectorStore:
    """Ingest documents into the vector store.

    Args:
    ----
        data_dir: Directory containing documents to ingest.
        db_dir: Directory to store the vector database.
        embedding_model_name: Name of the embedding model to use.

    Returns:
    -------
        ChromaVectorStore: The vector store with ingested documents.

    """
    print(f"Ingesting documents from {data_dir}...")

    # Create vector store directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)

    # Initialize the embedding model
    embedding_model = EmbeddingModel(model_name=embedding_model_name)

    # Create a Chroma client
    chroma_client = chromadb.PersistentClient(path=db_dir)
    collection_name = "rag_documents"

    # Create or get the collection
    try:
        chroma_client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
    except ValueError:
        print(f"Creating new collection: {collection_name}")

    # Initialize vector store
    vector_store = ChromaVectorStore(
        persist_directory=db_dir,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    # Load documents
    loader = DirectoryLoader(
        directory_path=data_dir,
        recursive=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Chunk documents
    chunker = RecursiveTextChunker(chunk_size=1000, chunk_overlap=200)
    chunked_documents = chunker.split_documents(documents)
    print(f"Created {len(chunked_documents)} chunks")

    # Extract content and metadata for vector store
    doc_contents = [doc["content"] for doc in chunked_documents]
    doc_metadatas = [doc["metadata"] for doc in chunked_documents]

    # Add documents to vector store
    vector_store.add_documents(documents=doc_contents, metadatas=doc_metadatas)
    print(f"Added {len(chunked_documents)} chunks to vector store")

    return vector_store


def run_interactive_mode(rag_pipeline: RAGPipeline) -> None:
    """Run the RAG system in interactive mode.

    Args:
    ----
        rag_pipeline: The configured RAG pipeline.

    """
    print("\nEntering interactive mode. Type 'exit' to quit.")

    while True:
        query = input("\nEnter your query: ")

        if query.lower() in ["exit", "quit", "q"]:
            break

        if not query.strip():
            continue

        result = rag_pipeline.query(query)

        print("\n" + "=" * 40)
        print("QUERY:", result["query"])
        print("=" * 40)
        print("ANSWER:", result["response"])
        print("=" * 40)
        print(f"Retrieved {len(result['retrieved_documents'])} documents:")

        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"\nDocument {i+1}:")
            print(f"  Source: {doc.get('metadata', {}).get('source', 'Unknown')}")
            print(f"  Content: {doc.get('content', '')[:100]}...")

        print("=" * 40)


def main() -> None:
    """Execute the main application logic.

    This function serves as the main entry point for the application.
    It parses command line arguments, sets up the RAG pipeline, and
    processes queries either in interactive mode or using a provided query.

    Returns
    -------
        None: This function doesn't return any value.

    """
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # If data directory is provided, ingest documents
    if args.data_dir:
        vector_store = ingest_documents(
            data_dir=args.data_dir,
            db_dir=args.db_dir,
            embedding_model_name=args.embedding_model,
        )
    else:
        # Load existing vector store
        if not os.path.exists(args.db_dir):
            print(f"Error: Vector database directory {args.db_dir} " "does not exist")
            print("Please provide a data directory to ingest documents first")
            sys.exit(1)

        # Initialize embedding model
        embedding_model = EmbeddingModel(model_name=args.embedding_model)

        # Create a Chroma client
        chroma_client = chromadb.PersistentClient(path=args.db_dir)
        collection_name = "rag_documents"

        try:
            chroma_client.get_collection(collection_name)
        except ValueError:
            print(f"Error: Collection {collection_name} does not exist")
            print("Please provide a data directory to ingest documents first")
            sys.exit(1)

        # Initialize vector store
        vector_store = ChromaVectorStore(
            persist_directory=args.db_dir,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

    # Initialize language model
    llm = LlamaCpp(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        verbose=True,
    )

    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        vectorstore=vector_store,
        llm=llm,
        top_k=args.top_k,
    )

    # Process query or run in interactive mode
    if args.query:
        result = rag_pipeline.query(args.query)

        print("\n" + "=" * 40)
        print("QUERY:", result["query"])
        print("=" * 40)
        print("ANSWER:", result["response"])
        print("=" * 40)
        print(f"Retrieved {len(result['retrieved_documents'])} documents:")

        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"\nDocument {i+1}:")
            print(f"  Source: {doc.get('metadata', {}).get('source', 'Unknown')}")
            print(f"  Content: {doc.get('content', '')[:100]}...")

        print("=" * 40)
    else:
        run_interactive_mode(rag_pipeline)


if __name__ == "__main__":
    main()
