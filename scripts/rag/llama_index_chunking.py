#!/usr/bin/env python
"""LlamaIndex Chunking for RAG System.

This script demonstrates sophisticated document chunking
and integration with the existing RAG system.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List

import chromadb

# Import LlamaIndex components
from llama_index_core import Document as LlamaDocument
from llama_index_core import StorageContext, VectorStoreIndex
from llama_index_core.extractors import KeywordExtractor, TitleExtractor
from llama_index_core.ingestion import IngestionPipeline
from llama_index_core.node_parser import SemanticSplitterNodeParser, TokenTextSplitter
from llama_index_embeddings_huggingface import HuggingFaceEmbedding
from llama_index_vector_stores_chroma import ChromaVectorStore

# Add the src directory to the path so we can import the llm_rag module
project_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to import from llm_rag
try:
    from llm_rag.document_processing.loaders import DirectoryLoader
except ImportError as e:
    logger.error(f"Could not import from llm_rag: {e}")
    sys.exit(1)


def load_documents_from_directory(docs_dir: str) -> List[Dict[str, Any]]:
    """Load documents from a directory using the existing llm_rag loader.

    Args:
        docs_dir: Path to the directory containing documents.

    Returns:
        List of document dictionaries.

    """
    logger.info(f"Loading documents from {docs_dir}")
    loader = DirectoryLoader(docs_dir)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def convert_to_llama_documents(documents: List[Dict[str, Any]]) -> List[LlamaDocument]:
    """Convert llm_rag documents to LlamaIndex documents.

    Args:
        documents: List of document dictionaries from llm_rag.

    Returns:
        List of LlamaIndex Document objects.

    """
    llama_docs = []
    for doc in documents:
        llama_doc = LlamaDocument(text=doc["content"], metadata=doc["metadata"])
        llama_docs.append(llama_doc)

    logger.info(f"Converted {len(llama_docs)} documents to LlamaIndex format")
    return llama_docs


def create_chunking_pipeline(
    chunk_size: int = 512, chunk_overlap: int = 50, use_semantic_chunking: bool = True
) -> IngestionPipeline:
    """Create a sophisticated document chunking pipeline using LlamaIndex.

    Args:
        chunk_size: Size of each chunk in tokens.
        chunk_overlap: Overlap between chunks in tokens.
        use_semantic_chunking: Whether to use semantic chunking.

    Returns:
        LlamaIndex IngestionPipeline for document processing.

    """
    # Create extractors for metadata enrichment
    extractors = [
        TitleExtractor(nodes=5),  # Extract titles from the first 5 nodes
        KeywordExtractor(keywords=10),  # Extract up to 10 keywords per node
    ]

    # Choose the appropriate chunking strategy
    if use_semantic_chunking:
        # Semantic chunking splits documents based on semantic meaning
        # For German text, using a multilingual model works better
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        embedding_model = HuggingFaceEmbedding(model_name=model_name)
        node_parser = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embedding_model
        )
        logger.info("Using semantic chunking with multilingual model")
    else:
        # Token-based chunking splits documents based on token count
        node_parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        msg = f"Using token-based chunking: size={chunk_size}, overlap={chunk_overlap}"
        logger.info(msg)

    # Create the ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            node_parser,  # First, parse documents into nodes
            *extractors,  # Then, extract metadata
        ]
    )

    return pipeline


def process_documents_with_llama_index(
    llama_docs: List[LlamaDocument],
    output_dir: str,
    collection_name: str = "llama_index_chunks",
    use_semantic_chunking: bool = True,
):
    """Process documents with LlamaIndex chunking and store in ChromaDB.

    Args:
        llama_docs: List of LlamaIndex Document objects.
        output_dir: Directory to store the ChromaDB.
        collection_name: Name of the ChromaDB collection.
        use_semantic_chunking: Whether to use semantic chunking.

    Returns:
        LlamaIndex VectorStoreIndex for querying.

    """
    # Create the chunking pipeline
    pipeline = create_chunking_pipeline(use_semantic_chunking=use_semantic_chunking)

    # Process documents through the pipeline
    nodes = pipeline.run(documents=llama_docs)
    logger.info(f"Created {len(nodes)} chunks from {len(llama_docs)} documents")

    # Print some sample chunks to show the difference
    for i, node in enumerate(nodes[:3]):
        logger.info(f"Sample chunk {i + 1}:")
        logger.info(f"  Text: {node.text[:100]}...")
        logger.info(f"  Metadata: {node.metadata}")

    # Set up ChromaDB for storage
    os.makedirs(output_dir, exist_ok=True)

    # Use a multilingual embedding model for German text
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_model = HuggingFaceEmbedding(model_name=model_name)

    # Create ChromaDB vector store
    chroma_client = chromadb.PersistentClient(path=output_dir)
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index from nodes
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)

    logger.info(f"Successfully indexed {len(nodes)} chunks in ChromaDB: {output_dir}")

    # Return the index for querying
    return index


def integrate_with_existing_rag(index, query: str, similarity_top_k: int = 5):
    """Demonstrate how to integrate LlamaIndex with the existing RAG system.

    Args:
        index: LlamaIndex VectorStoreIndex.
        query: Query string.
        similarity_top_k: Number of similar documents to retrieve.

    """
    # Create a retriever from the index
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    # Retrieve relevant nodes
    retrieved_nodes = retriever.retrieve(query)

    logger.info(f"Retrieved {len(retrieved_nodes)} nodes for query: '{query}'")

    # Print retrieved nodes
    for i, node in enumerate(retrieved_nodes):
        logger.info(f"Node {i + 1} (Score: {node.score}):")
        logger.info(f"  Text: {node.node.text[:150]}...")
        logger.info(f"  Metadata: {node.node.metadata}")

    # Here you would integrate with your existing RAG pipeline
    # For example:
    # 1. Convert nodes to the format your RAG system expects
    # 2. Create a context string from the nodes
    # 3. Pass the context to your LLM


def main():
    """Run LlamaIndex chunking demonstration."""
    parser = argparse.ArgumentParser(description="Demonstrate LlamaIndex chunking for RAG system")
    parser.add_argument(
        "--docs-dir", type=str, default="data/documents/test_subset", help="Directory containing documents to process"
    )
    parser.add_argument("--output-dir", type=str, default="llama_index_db", help="Directory to store the ChromaDB")
    parser.add_argument(
        "--collection-name", type=str, default="llama_index_chunks", help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--semantic-chunking", action="store_true", help="Use semantic chunking instead of token-based chunking"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Welche technischen Anforderungen gelten für DIN Standards?",
        help="Query to test retrieval (default is in German)",
    )

    args = parser.parse_args()

    # Load documents using existing loader
    documents = load_documents_from_directory(args.docs_dir)

    # Convert to LlamaIndex documents
    llama_docs = convert_to_llama_documents(documents)

    # Process with LlamaIndex chunking
    try:
        index = process_documents_with_llama_index(
            llama_docs, args.output_dir, args.collection_name, args.semantic_chunking
        )

        # Test retrieval
        integrate_with_existing_rag(index, args.query)

    except ImportError as e:
        error_msg = (
            f"Error: {e}. Make sure all required packages are installed.\n"
            "Try: pip install chromadb llama-index-vector-stores-chroma "
            "llama-index-embeddings-huggingface"
        )
        logger.error(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
