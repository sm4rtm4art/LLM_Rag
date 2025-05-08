"""Main module for the llm-rag package.

This module provides the main entry point for the llm-rag application.
It demonstrates how to set up and use a simple RAG pipeline.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Union, cast

import chromadb

# Import necessary callback handlers for LlamaCpp
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    StreamingStdOutCallbackHandler,
)
from langchain_core.language_models.llms import LLM
from langchain_core.vectorstores.base import VectorStore as LangchainVectorStore
from llama_cpp import Llama  # type: ignore

# Use relative imports
from .document_processing.chunking import RecursiveTextChunker
from .document_processing.loaders.directory_loader import DirectoryLoader
from .models.embeddings import EmbeddingModel
from .rag.pipeline import RAGPipeline
from .vectorstore.chroma import ChromaVectorStore


# Create a custom wrapper if needed
class CustomLlamaCpp(LLM):
    """Custom wrapper for the Llama model from llama-cpp-python.

    This class provides a LangChain compatible interface to the Llama model.
    """

    model_path: str
    callbacks: List[Any] = []
    model: Any = None

    def __init__(self, model_path: str, callbacks: Optional[List[Any]] = None, **kwargs: Any) -> None:
        """Initialize the CustomLlamaCpp.

        Args:
        ----
            model_path: Path to the Llama model file
            callbacks: List of callback handlers
            **kwargs: Additional parameters to pass to the Llama model

        """
        # Initialize the parent class
        super().__init__(callbacks=callbacks)
        self.model_path = model_path
        self.callbacks = callbacks or []

        # Only try to load the model if the file exists
        # This helps with testing
        if os.path.exists(model_path):
            self.model = Llama(model_path=model_path, **kwargs)
        else:
            print(f'Warning: Model file {model_path} does not exist. Running in test mode.')
            self.model = None

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return 'custom_llama_cpp'

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Llama model with the given prompt.

        Args:
        ----
            prompt: The prompt to send to the model
            stop: A list of strings to stop generation when encountered
            run_manager: Callback manager for the LLM run
            **kwargs: Additional parameters for the model call

        Returns:
        -------
            str: The model's response

        """
        if self.model is None:
            # Return a dummy response for testing
            return 'Test response from CustomLlamaCpp'

        # Call the model with the prompt
        response = self.model(prompt=prompt, **kwargs)
        return str(response['choices'][0]['text'])


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.

    """
    parser = argparse.ArgumentParser(description='RAG system for document retrieval and question answering.')

    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing documents to ingest',
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Query to run against the RAG system. If not provided, interactive mode is used.',
    )
    parser.add_argument(
        '--db-dir',
        type=str,
        default='./chroma_db',
        help='Directory for the vector database (default: ./chroma_db)',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='./models/llama-2-7b-chat.gguf',
        help='Path to the LLM model (default: ./models/llama-2-7b-chat.gguf)',
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Name of the embedding model (default: all-MiniLM-L6-v2)',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=4,
        help='Number of documents to retrieve (default: 4)',
    )
    parser.add_argument(
        '--n-gpu-layers',
        type=int,
        default=0,
        help='Number of GPU layers to use (default: 0, CPU only)',
    )
    parser.add_argument(
        '--n-ctx',
        type=int,
        default=4096,
        help='Context size for the LLM (default: 4096)',
    )
    parser.add_argument(
        '--create-empty',
        action='store_true',
        help="Create an empty collection if one doesn't exist",
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
    print(f'Ingesting documents from {data_dir}...')

    # Create vector store directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)

    # Initialize embedding model
    model_name = embedding_model_name
    if not model_name.startswith('sentence-transformers/') and '/' not in model_name:
        model_name = f'sentence-transformers/{model_name}'

    embedding_model = EmbeddingModel(model_name=model_name)

    # Create a Chroma client
    chroma_client = chromadb.PersistentClient(path=db_dir)
    collection_name = 'rag_documents'

    # Create or get the collection
    try:
        chroma_client.get_collection(collection_name)
        print(f'Using existing collection: {collection_name}')
    except ValueError:
        print(f'Creating new collection: {collection_name}')

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
    print(f'Loaded {len(documents)} documents')

    # Chunk documents
    chunker = RecursiveTextChunker(chunk_size=1000, chunk_overlap=200)
    chunked_documents = chunker.split_documents(documents)
    print(f'Created {len(chunked_documents)} chunks')

    # Process documents to ensure correct format
    doc_contents: List[str] = []
    doc_metadatas: List[Dict[str, Any]] = []

    # Process each document to extract content and metadata
    for i, doc in enumerate(chunked_documents):
        try:
            content: str = ''
            metadata: Dict[str, Any] = {}

            if isinstance(doc, dict):
                content = cast(str, doc.get('content', ''))
                metadata = cast(Dict[str, Any], doc.get('metadata', {}))
            elif hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):  # type: ignore[unreachable]
                # Handle Document objects
                content = cast(str, getattr(doc, 'page_content', ''))
                metadata = cast(Dict[str, Any], getattr(doc, 'metadata', {}))

            # Only add non-empty content
            if content:
                doc_contents.append(content)
                doc_metadatas.append(metadata)
        except Exception as e:
            print(f'Error processing document {i}: {e}')
            continue

    # Add documents to vector store - ensure all contents are strings
    # and metadata is properly formatted
    str_contents = [str(content) for content in doc_contents]

    # Ensure all metadata is a dictionary with string keys and valid values
    formatted_metadatas: List[Dict[str, Union[str, bool, int, float]]] = []
    for metadata in doc_metadatas:
        # Convert all values to strings except booleans, ints, and floats
        formatted_metadata: Dict[str, Union[str, bool, int, float]] = {}
        for k, v in metadata.items():
            if isinstance(v, (bool, int, float)):
                formatted_metadata[str(k)] = v
            else:
                formatted_metadata[str(k)] = str(v)
        formatted_metadatas.append(formatted_metadata)

    # Add documents to vector store
    vector_store.add_documents(documents=str_contents, metadatas=formatted_metadatas)
    print(f'Added {len(str_contents)} chunks to vector store')

    return vector_store


def run_interactive_mode(rag_pipeline: RAGPipeline) -> None:
    """Run the RAG system in interactive mode.

    Args:
    ----
        rag_pipeline: The configured RAG pipeline.

    """
    print("\nEntering interactive mode. Type 'exit' to quit.")

    while True:
        query = input('\nEnter your query: ')

        if query.lower() in ['exit', 'quit', 'q']:
            break

        if not query.strip():
            continue

        result = rag_pipeline.query(query)

        print('\n' + '=' * 40)
        print('QUERY:', result['query'])
        print('=' * 40)
        print('ANSWER:', result['response'])
        print('=' * 40)
        print(f'Retrieved {len(result["documents"])} documents:')

        for i, doc in enumerate(result['documents']):
            print(f'\nDocument {i + 1}:')
            print(f'  Source: {doc.get("metadata", {}).get("source", "Unknown")}')
            print(f'  Content: {doc.get("content", "")[:100]}...')

        print('=' * 40)


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
            print(f'Error: Vector database directory {args.db_dir} does not exist')
            print('Please provide a data directory to ingest documents first')
            sys.exit(1)

        # Initialize embedding model
        model_name = args.embedding_model
        if not model_name.startswith('sentence-transformers/') and '/' not in model_name:
            model_name = f'sentence-transformers/{model_name}'

        embedding_model = EmbeddingModel(model_name=model_name)

        # Create a Chroma client
        chroma_client = chromadb.PersistentClient(path=args.db_dir)
        collection_name = 'rag_documents'

        try:
            chroma_client.get_collection(collection_name)
        except (ValueError, chromadb.errors.InvalidCollectionException):
            if args.create_empty:
                print(f'Creating empty collection: {collection_name}')
                chroma_client.create_collection(collection_name)
            else:
                print(f'Error: Collection {collection_name} does not exist in {args.db_dir}')
                print('Please provide a data directory to ingest documents using --data-dir')
                print('Or use --create-empty to create an empty collection')
                print('Example: python -m src.llm_rag.main --data-dir ./data/documents')
                sys.exit(1)

        # Initialize vector store
        vector_store = ChromaVectorStore(
            persist_directory=args.db_dir,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

    # Initialize language model
    # Set up callbacks for streaming output
    callbacks = [StreamingStdOutCallbackHandler()]

    # Use CustomLlamaCpp instead of ChatLlamaCpp
    llm = CustomLlamaCpp(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        verbose=True,
        f16_kv=True,
        n_batch=512,
        callbacks=callbacks,
    )

    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        vectorstore=cast(LangchainVectorStore, vector_store),
        llm=llm,
        top_k=args.top_k,
    )

    # Process query or run in interactive mode
    if args.query:
        result = rag_pipeline.query(args.query)

        print('\n' + '=' * 40)
        print('QUERY:', result['query'])
        print('=' * 40)
        print('ANSWER:', result['response'])
        print('=' * 40)
        print(f'Retrieved {len(result["documents"])} documents:')

        for i, doc in enumerate(result['documents']):
            print(f'\nDocument {i + 1}:')
            print(f'  Source: {doc.get("metadata", {}).get("source", "Unknown")}')
            print(f'  Content: {doc.get("content", "")[:100]}...')

        print('=' * 40)
    else:
        run_interactive_mode(rag_pipeline)


if __name__ == '__main__':
    main()
