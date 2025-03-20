#!/usr/bin/env python
"""End-to-End Test for RAG Pipeline with Anti-Hallucination..

This script demonstrates a complete workflow:
1. Load existing documents from data/documents/test_subset
2. Create or load a vector store with ChromaDB
3. Set up a RAG pipeline with Llama 3.3
4. Enable anti-hallucination features
5. Process queries and analyze results
"""

import argparse
import logging
import os
import time
from typing import Dict, List, Optional

from src.llm_rag.document_processing.loaders import DirectoryLoader
from src.llm_rag.document_processing.processors import DocumentProcessor, TextSplitter
from src.llm_rag.models.factory import ModelBackend, ModelFactory
from src.llm_rag.rag.anti_hallucination import HallucinationConfig
from src.llm_rag.rag.anti_hallucination.entity import extract_key_entities
from src.llm_rag.rag.anti_hallucination.processing import post_process_response
from src.llm_rag.rag.pipeline import RAGPipeline
from src.llm_rag.rag.pipeline.pipeline_builder import RAGPipelineBuilder
from src.llm_rag.vectorstore.chroma import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_documents(docs_dir: str) -> List[Dict]:
    """Load documents from the specified directory.

    Args:
        docs_dir: Path to the directory containing documents

    Returns:
        List of loaded documents

    """
    logger.info(f"Loading documents from {docs_dir}")

    # Create directory loader
    loader = DirectoryLoader(docs_dir)

    # Load documents
    documents = loader.load()

    logger.info(f"Loaded {len(documents)} documents")

    # Print document summary
    for i, doc in enumerate(documents[:3], start=1):
        metadata = doc.get("metadata", {})
        source = metadata.get("source", "Unknown")
        content_len = len(doc.get("content", ""))
        logger.info(f"Document {i}: {source} ({content_len} characters)")

    if len(documents) > 3:
        logger.info(f"... and {len(documents) - 3} more documents")

    return documents


def process_documents(documents: List[Dict]) -> List[Dict]:
    """Process documents by splitting them into chunks.

    Args:
        documents: List of documents to process

    Returns:
        List of processed document chunks

    """
    logger.info("Processing documents")

    # Create text splitter
    text_splitter = TextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # Create document processor with text splitter
    processor = DocumentProcessor(text_splitter=text_splitter)

    # Process documents
    processed_docs = processor.process(documents)

    logger.info(f"Created {len(processed_docs)} document chunks")

    return processed_docs


def build_vectorstore(
    documents: List[Dict],
    output_dir: str,
    collection_name: str = "documents",
) -> ChromaVectorStore:
    """Build a vector store from the processed documents.

    Args:
        documents: List of processed documents
        output_dir: Directory to store the vector database
        collection_name: Name of the collection in ChromaDB

    Returns:
        Initialized vector store

    """
    logger.info(f"Building vector store in {output_dir}")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize vector store
    vectorstore = ChromaVectorStore(
        persist_directory=output_dir,
        collection_name=collection_name,
    )

    # Extract contents and metadata
    doc_contents = []
    doc_metadatas = []

    for doc in documents:
        if isinstance(doc, dict) and "content" in doc:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Clean metadata - ensure all values are strings, ints, floats, or bools
            clean_metadata = {}
            for k, v in metadata.items():
                if v is None:
                    # Skip None values
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                else:
                    # Convert other types to string
                    clean_metadata[k] = str(v)

            if content and isinstance(content, str):
                doc_contents.append(content)
                doc_metadatas.append(clean_metadata)

    # Add documents to vector store
    if doc_contents:
        logger.info(f"Adding {len(doc_contents)} documents to vector store")
        vectorstore.add_documents(documents=doc_contents, metadatas=doc_metadatas)
    else:
        logger.warning("No valid documents to add to vector store")

    logger.info(f"Vector store built with {vectorstore.get_collection_size()} documents")

    return vectorstore


def load_vectorstore(
    output_dir: str,
    collection_name: str = "documents",
) -> ChromaVectorStore:
    """Load an existing vector store.

    Args:
        output_dir: Directory containing the vector database
        collection_name: Name of the collection in ChromaDB

    Returns:
        Loaded vector store

    """
    logger.info(f"Loading vector store from {output_dir}")

    # Initialize vector store from existing data
    vectorstore = ChromaVectorStore(
        persist_directory=output_dir,
        collection_name=collection_name,
    )

    # Get collection stats
    collection_size = vectorstore.get_collection_size()
    logger.info(f"Vector store loaded with {collection_size} documents")

    return vectorstore


def setup_rag_pipeline(
    vectorstore: ChromaVectorStore,
    model_name: str = "llama3",
    top_k: int = 5,
    apply_anti_hallucination: bool = True,
) -> RAGPipeline:
    """Set up a RAG pipeline with anti-hallucination features.

    Args:
        vectorstore: Vector store for document retrieval
        model_name: Name of the Ollama model to use
        top_k: Number of documents to retrieve
        apply_anti_hallucination: Whether to apply anti-hallucination

    Returns:
        Configured RAG pipeline

    """
    logger.info(f"Setting up RAG pipeline with Ollama model: {model_name}")

    # Create model factory
    factory = ModelFactory()

    # Initialize language model
    try:
        llm = factory.create_model(
            model_path_or_name=model_name,
            backend=ModelBackend.OLLAMA,
        )
        logger.info(f"Successfully initialized Ollama model: {model_name}")
    except Exception as e:
        logger.error(f"Error initializing Ollama model: {e}")
        raise RuntimeError(f"Failed to initialize Ollama model: {e}") from e

    # Create RAG pipeline using the builder pattern
    builder = RAGPipelineBuilder()
    pipeline = (
        builder.with_retriever(vectorstore, top_k=top_k)
        .with_default_formatter(max_context_length=4000, include_metadata=True)
        .with_llm_generator(llm=llm, apply_anti_hallucination=apply_anti_hallucination)
        .build()
    )

    logger.info("RAG pipeline created successfully")

    return pipeline


def analyze_response(response: str, context: str) -> None:
    """Analyze a response for potential hallucinations.

    Args:
        response: Generated response
        context: Retrieved context used for generation

    """
    print("\n===== RESPONSE ANALYSIS =====")

    # Extract entities from response and context
    response_entities = extract_key_entities(response)
    context_entities = extract_key_entities(context)

    # Identify missing entities (potential hallucinations)
    missing_entities = [entity for entity in response_entities if entity not in context_entities]

    # Calculate coverage ratio
    if response_entities:
        coverage_ratio = 1.0 - (len(missing_entities) / len(response_entities))
    else:
        coverage_ratio = 1.0

    # Print analysis results
    print(f"Entity coverage: {coverage_ratio:.2f}")
    print(f"Entities in response: {len(response_entities)}")
    print(f"Entities in context: {len(context_entities)}")
    print(f"Potentially hallucinated entities: {len(missing_entities)}")

    if missing_entities:
        print("\nEntities not found in context:")
        for entity in missing_entities[:10]:
            print(f"  - {entity}")
        if len(missing_entities) > 10:
            print(f"  ... and {len(missing_entities) - 10} more")

    print("===============================\n")


def interactive_rag_session(
    pipeline: RAGPipeline,
    anti_hallucination_config: Optional[HallucinationConfig] = None,
) -> None:
    """Run an interactive RAG session.

    Args:
        pipeline: Configured RAG pipeline
        anti_hallucination_config: Configuration for anti-hallucination

    """
    print("\n==== RAG Interactive Session ====")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'toggle' to toggle anti-hallucination")
    print("===============================")

    # Default anti-hallucination config
    if anti_hallucination_config is None:
        anti_hallucination_config = HallucinationConfig(
            entity_threshold=0.7,
            embedding_threshold=0.5,
            flag_for_human_review=True,
            use_embeddings=True,
        )

    use_anti_hallucination = True

    while True:
        # Get user query
        query = input("\nEnter your query (or 'exit' to quit): ")

        if query.lower() in ("exit", "quit"):
            break

        if query.lower() == "toggle":
            use_anti_hallucination = not use_anti_hallucination
            print(f"\nAnti-hallucination features: {'ON' if use_anti_hallucination else 'OFF'}")
            continue

        # Process the query
        print("\nProcessing query...")
        start_time = time.time()

        try:
            # Query the RAG pipeline
            result = pipeline.query(query)

            # Extract response and context
            if isinstance(result, dict):
                response = result.get("response", "No response generated")
                context = result.get("context", "")
            else:
                response = result
                context = ""

            # Apply post-processing if enabled
            if use_anti_hallucination and context:
                processed_response, metadata = post_process_response(
                    response=response,
                    context=context,
                    config=anti_hallucination_config,
                    return_metadata=True,
                )
                # Print the processed response
                print("\n===== PROCESSED RESPONSE =====")
                print(processed_response)

                # Analyze the response
                analyze_response(response, context)
            else:
                # Print the raw response
                print("\n===== RAW RESPONSE =====")
                print(response)

            # Print time taken
            elapsed_time = time.time() - start_time
            print(f"\nQuery processed in {elapsed_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\nError: {str(e)}")


def main():
    """Run the main function."""
    parser = argparse.ArgumentParser(description="End-to-End RAG Pipeline Test")

    parser.add_argument(
        "--docs-dir",
        type=str,
        default="data/documents/test_subset",
        help="Directory containing documents",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="chroma_db_test",
        help="Directory to store the vector database",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3",
        help="Ollama model name (default: llama3)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building the vector store if it exists",
    )
    parser.add_argument(
        "--no-anti-hallucination",
        action="store_true",
        help="Disable anti-hallucination features",
    )

    args = parser.parse_args()

    try:
        # Check if we need to build the vector store
        if not args.skip_build or not os.path.exists(args.output_dir):
            # Load documents
            documents = load_documents(args.docs_dir)

            # Process documents
            processed_docs = process_documents(documents)

            # Build vector store
            vectorstore = build_vectorstore(
                documents=processed_docs,
                output_dir=args.output_dir,
            )
        else:
            # Load existing vector store
            logger.info(f"Using existing vector store at {args.output_dir}")
            vectorstore = load_vectorstore(args.output_dir)

        # Set up RAG pipeline
        pipeline = setup_rag_pipeline(
            vectorstore=vectorstore,
            model_name=args.model,
            top_k=args.top_k,
            apply_anti_hallucination=not args.no_anti_hallucination,
        )

        # Run interactive session
        interactive_rag_session(pipeline)

    except Exception as e:
        logger.error(f"Error during RAG testing: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
