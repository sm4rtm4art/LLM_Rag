#!/usr/bin/env python3
"""Demo script for multi-modal RAG with DIN standards.

This script demonstrates how to use the multi-modal RAG system to process
DIN standards with text, tables, and images/drawings.
"""

import argparse

# Configure logging
import logging
import sys
from pathlib import Path

import torch

try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    # Fallback import path for older versions
    from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_model(model_name_or_path, device='auto', max_length=2048):
    """Load a Hugging Face model for text generation."""
    logger.info(f'Loading model: {model_name_or_path}')

    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )

    # Create text generation pipeline
    text_generation_pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
    )

    # Create LangChain wrapper
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return llm


def load_documents(doc_path, extract_tables=True, extract_images=True):
    """Load documents from a path."""
    logger.info(f'Loading documents from: {doc_path}')

    path = Path(doc_path)
    documents = []

    if path.is_file():
        # Load a single file
        if path.suffix.lower() == '.pdf':
            # For PDF files, try to use enhanced loader if available
            try:
                from llm_rag.document_processing.loaders import EnhancedPDFLoader

                loader = EnhancedPDFLoader(
                    file_path=path,
                    extract_tables=extract_tables,
                    extract_images=extract_images,
                )
            except (ImportError, AttributeError):
                # Fall back to standard PDF loader
                from llm_rag.document_processing.loaders import PDFLoader

                loader = PDFLoader(file_path=path)
        elif path.suffix.lower() in ['.txt', '.md', '.html']:
            # For text files
            from llm_rag.document_processing.loaders import TextFileLoader

            loader = TextFileLoader(file_path=path)
        else:
            logger.error(f'Unsupported file type: {path.suffix}')
            sys.exit(1)

        documents = loader.load()
    elif path.is_dir():
        # Load all files in the directory
        from llm_rag.document_processing.loaders import DirectoryLoader

        loader = DirectoryLoader(
            directory_path=path,
            recursive=True,
        )
        documents = loader.load()
    else:
        logger.error(f'Invalid path: {doc_path}')
        sys.exit(1)

    logger.info(f'Loaded {len(documents)} document chunks')

    # Count document types
    doc_types: dict[str, int] = {}
    for doc in documents:
        filetype = doc.get('metadata', {}).get('filetype', 'unknown')
        doc_types[filetype] = doc_types.get(filetype, 0) + 1

    logger.info(f'Document types: {doc_types}')

    return documents


def process_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Process and chunk documents for the RAG system."""
    logger.info('Processing documents')

    # Create chunker
    try:
        # Try to use MultiModalChunker if available
        from llm_rag.document_processing.chunking import MultiModalChunker

        chunker = MultiModalChunker(
            text_chunk_size=chunk_size,
            text_chunk_overlap=chunk_overlap,
            preserve_images=True,
            preserve_tables=True,
        )
    except (ImportError, AttributeError):
        # Fall back to RecursiveTextChunker
        from llm_rag.document_processing.chunking import RecursiveTextChunker

        chunker = RecursiveTextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Process documents
    chunked_docs = chunker.split_documents(documents)
    logger.info(f'Created {len(chunked_docs)} chunks')

    return chunked_docs


def create_vectorstore(
    documents,
    persist_directory='chroma_db',
    collection_name='test_collection',
):
    """Create a vector store from processed documents."""
    logger.info('Creating vector store')

    try:
        # Try to use MultiModalVectorStore if available
        from llm_rag.vectorstore.multimodal import MultiModalVectorStore

        vectorstore = MultiModalVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    except (ImportError, AttributeError):
        # Fall back to ChromaVectorStore
        from llm_rag.vectorstore.chroma import ChromaVectorStore

        vectorstore = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    # Extract content and metadata
    texts = []
    metadatas = []
    for doc in documents:
        texts.append(doc.get('content', ''))
        metadatas.append(doc.get('metadata', {}))

    # Add documents to vector store
    vectorstore.add_documents(texts, metadatas)
    logger.info(f'Added {len(texts)} documents to vector store')

    return vectorstore


def create_rag_pipeline(vectorstore, llm, top_k=3):
    """Create a RAG pipeline."""
    logger.info('Creating RAG pipeline')

    try:
        # Try to use MultiModalRAGPipeline if available
        from llm_rag.rag import MultiModalRAGPipeline

        pipeline = MultiModalRAGPipeline(
            vectorstore=vectorstore,
            llm=llm,
            top_k=top_k,
        )
    except (ImportError, AttributeError):
        # Fall back to ConversationalRAGPipeline
        from llm_rag.rag import ConversationalRAGPipeline

        # Create the pipeline with appropriate parameters
        pipeline = ConversationalRAGPipeline(
            retriever=vectorstore.as_retriever(search_kwargs={'k': top_k}),
            llm_chain=llm,
            top_k=top_k,
        )

    return pipeline


def interactive_query(rag_pipeline):
    """Run an interactive query session."""
    logger.info('Starting interactive query session')
    print('\n=== Document RAG Demo ===')
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'reset' to reset the conversation history")

    while True:
        # Get user query
        query = input('\nEnter your query: ')
        query = query.strip()

        # Check for exit command
        if query.lower() in ['exit', 'quit']:
            break

        # Check for reset command
        if query.lower() == 'reset':
            rag_pipeline.reset_memory()
            print('Conversation history reset')
            continue

        # Process query
        try:
            result = rag_pipeline.query(query)
            response = result.get('response', '')
            retrieved_docs = result.get('retrieved_documents', [])

            # Print response
            print('\n=== Response ===')
            print(response)

            # Print sources (optional)
            if retrieved_docs:
                print('\n=== Sources ===')
                for i, doc in enumerate(retrieved_docs[:3]):
                    filetype = doc.get('metadata', {}).get('filetype', 'unknown')
                    source = doc.get('metadata', {}).get('source', 'unknown')
                    print(f'[{i + 1}] {filetype.upper()} - {source}')

        except Exception as e:
            logger.error(f'Error processing query: {str(e)}')
            print(f'Error: {str(e)}')


def main():
    """Run the document RAG demo."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Document RAG Demo')
    parser.add_argument(
        '--doc_path',
        type=str,
        required=True,
        help='Path to document or directory containing documents',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='microsoft/phi-2',
        help='Name or path of the model to use',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cpu', 'cuda', 'auto'],
        help='Device to use for model inference',
    )
    parser.add_argument(
        '--persist_dir',
        type=str,
        default='chroma_db',
        help='Directory to persist the vector store',
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Number of documents to retrieve',
    )
    parser.add_argument(
        '--no_tables',
        action='store_true',
        help='Disable table extraction',
    )
    parser.add_argument(
        '--no_images',
        action='store_true',
        help='Disable image extraction',
    )

    args = parser.parse_args()

    # Load model
    llm = load_model(args.model, args.device)

    # Load documents
    documents = load_documents(
        args.doc_path,
        extract_tables=not args.no_tables,
        extract_images=not args.no_images,
    )

    # Process documents
    processed_docs = process_documents(documents)

    # Create vector store
    vectorstore = create_vectorstore(
        processed_docs,
        persist_directory=args.persist_dir,
    )

    # Create RAG pipeline
    rag_pipeline = create_rag_pipeline(
        vectorstore,
        llm,
        top_k=args.top_k,
    )

    # Run interactive query session
    interactive_query(rag_pipeline)


if __name__ == '__main__':
    main()
