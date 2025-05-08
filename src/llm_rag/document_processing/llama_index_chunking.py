#!/usr/bin/env python
"""Process documents using LlamaIndex chunking for the RAG system.

This script demonstrates sophisticated document chunking
and integration with the existing RAG system.
"""

import logging
from typing import Any, Dict, List

# Import LlamaIndex components with proper handling of import errors
try:
    # Core document class
    from llama_index_core import Document as LlamaDocument

    # Extractors for metadata enhancement
    from llama_index_core.extractors import KeywordExtractor, TitleExtractor

    # Pipeline for document processing
    from llama_index_core.ingestion import IngestionPipeline

    # Node parsers for different chunking strategies
    from llama_index_core.node_parser import SemanticSplitterNodeParser, TokenTextSplitter
except ImportError as err:
    raise ImportError('LlamaIndex packages not found. Install with: pip install llama-index-core') from err


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Process documents using LlamaIndex chunking."""
    logger.info('LlamaIndex document processing utility')
    logger.info('Use this module for LlamaIndex chunking')


def convert_to_llama_documents(documents: List[Dict[str, Any]]) -> List[LlamaDocument]:
    """Convert our document format to LlamaIndex Document objects.

    Args:
        documents: List of document dictionaries with content and metadata.

    Returns:
        List of LlamaIndex Document objects.

    """
    llama_docs = []
    for doc in documents:
        # Create a LlamaIndex Document from our document format
        llama_doc = LlamaDocument(text=doc.get('content', ''), metadata=doc.get('metadata', {}))
        llama_docs.append(llama_doc)

    logger.info(f'Converted {len(llama_docs)} documents to LlamaIndex format')
    return llama_docs


def create_chunking_pipeline(
    use_semantic_chunking: bool = True,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> IngestionPipeline:
    """Create a LlamaIndex ingestion pipeline with chunking.

    Args:
        use_semantic_chunking: Use semantic vs token chunking
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        LlamaIndex IngestionPipeline object

    """
    # Create transformations/extractors
    transformations = [
        TitleExtractor(),
        KeywordExtractor(keywords_per_chunk=6),
    ]

    # Create the appropriate node parser based on chunking strategy
    if use_semantic_chunking:
        # Semantic chunking based on content meaning
        node_parser = SemanticSplitterNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info('Using semantic chunking strategy')
    else:
        # Token-based chunking for more predictable chunk sizes
        node_parser = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.info('Using token-based chunking strategy')

    # Create and return the pipeline
    pipeline = IngestionPipeline(
        transformations=transformations,
        node_parser=node_parser,
    )

    return pipeline


def process_documents(
    documents: List[Dict[str, Any]],
    use_semantic_chunking: bool = True,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """Process documents using LlamaIndex chunking.

    Args:
        documents: Document list with content and metadata
        use_semantic_chunking: Use semantic vs token chunking
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks with updated metadata

    """
    # Convert to LlamaIndex documents
    llama_docs = convert_to_llama_documents(documents)

    # Create the chunking pipeline
    pipeline = create_chunking_pipeline(
        use_semantic_chunking=use_semantic_chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Process documents through the pipeline
    nodes = pipeline.run(documents=llama_docs)
    logger.info(f'Created {len(nodes)} chunks from {len(llama_docs)} documents')

    # Convert nodes back to our document format
    result_chunks = []
    for i, node in enumerate(nodes):
        # Create metadata with chunk information
        metadata = node.metadata.copy() if node.metadata else {}
        metadata.update(
            {
                'chunk_index': i,
                'chunk_count': len(nodes),
                'is_llama_index': True,
            }
        )

        # Add any extracted info like title and keywords
        if hasattr(node, 'excluded_embed_metadata_keys'):
            for key in node.excluded_embed_metadata_keys:
                if key in node.metadata and key not in metadata:
                    metadata[key] = node.metadata[key]

        # Create document chunk
        chunk = {
            'content': node.text,
            'metadata': metadata,
        }
        result_chunks.append(chunk)

    return result_chunks


if __name__ == '__main__':
    main()
