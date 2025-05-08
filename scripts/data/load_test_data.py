#!/usr/bin/env python
"""Load test data into Chroma database for RAG testing.

This script loads sample documents into a Chroma vector database
to test the RAG system with anti-hallucination features.
"""

import logging
import os
import sys
from typing import Dict, List

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print('Error: Required dependencies not found. Try installing langchain packages.')
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample documents about RAG, anti-hallucination, and related topics
SAMPLE_TEXTS = [
    {
        'title': 'Introduction to RAG',
        'content': """
        Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by retrieving
        relevant information from external knowledge sources before generating responses. RAG combines the strengths
        of retrieval-based and generation-based approaches to AI.

        The key components of a RAG system include:
        1. A vector database for storing document embeddings
        2. An embedding model for converting text to vector representations
        3. A retrieval mechanism to find relevant documents
        4. A language model to generate responses based on the retrieved context

        RAG helps address the limitations of traditional language models by providing access to specific,
        up-to-date information that may not be present in the model's training data.
        """,
    },
    {
        'title': 'Anti-Hallucination Techniques',
        'content': """
        Hallucination in language models refers to the generation of content that is factually incorrect or
        not supported by the provided context. Anti-hallucination techniques aim to reduce these issues.

        Common anti-hallucination approaches include:
        - Entity verification: Checking if entities mentioned in the response appear in the source context
        - Semantic similarity: Measuring the embedding similarity between response and context
        - Constrained decoding: Limiting the model's generation to only use information from the context
        - Human review flagging: Identifying potentially hallucinated content for expert review

        These techniques can be combined for more robust hallucination detection and mitigation.
        """,
    },
    {
        'title': 'Vector Databases',
        'content': """
        Vector databases are specialized storage systems designed to efficiently store, index, and query
        high-dimensional vector embeddings. They are a critical component of modern RAG systems.

        Popular vector databases include:
        - Chroma: An open-source embedding database
        - Pinecone: A fully managed vector database service
        - Weaviate: A vector search engine with semantic search capabilities
        - FAISS: Facebook AI Similarity Search for efficient similarity search

        These databases use various algorithms like HNSW (Hierarchical Navigable Small World) for
        approximate nearest neighbor search, enabling fast retrieval of similar vectors.
        """,
    },
    {
        'title': 'Embedding Models',
        'content': """
        Embedding models convert text into numerical vector representations that capture semantic meaning.
        These vectors allow machines to understand relationships between different pieces of text.

        Common embedding models include:
        - OpenAI's text-embedding-ada-002
        - Sentence Transformers like all-MiniLM-L6-v2
        - BERT and its variants
        - E5 and other specialized embedding models

        The quality of embeddings significantly impacts RAG system performance, as better embeddings lead
        to more relevant document retrieval.
        """,
    },
    {
        'title': 'DIN Standards',
        'content': """
        DIN standards are technical standards established by the German Institute for Standardization
        (Deutsches Institut für Normung). These standards cover various fields including engineering,
        manufacturing, and information technology.

        Important aspects of DIN standards:
        - DIN EN ISO 9001: Quality management systems requirements
        - DIN 5008: Standards for letter formatting and text processing
        - DIN 476: Paper sizes (basis for ISO 216, which defines A4)

        DIN standards are widely used throughout Europe and internationally, often serving as the basis
        for ISO standards. They provide important guidelines for ensuring quality, safety, and compatibility
        across industries.
        """,
    },
]


def create_documents(texts: List[Dict[str, str]]) -> List[Document]:
    """Create Document objects from text dictionaries.

    Args:
        texts: List of dictionaries with title and content

    Returns:
        List of Document objects

    """
    documents = []
    for item in texts:
        doc = Document(page_content=item['content'], metadata={'title': item['title']})
        documents.append(doc)

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks.

    Args:
        documents: List of Document objects

    Returns:
        List of split Document objects

    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=['\n\n', '\n', '. ', ' ', '']
    )

    split_docs = text_splitter.split_documents(documents)
    logger.info(f'Split {len(documents)} documents into {len(split_docs)} chunks')

    return split_docs


def load_documents_to_chroma(
    documents: List[Document], persist_directory: str, embedding_model_name: str = 'all-MiniLM-L6-v2'
) -> Chroma:
    """Load documents into a Chroma vector database.

    Args:
        documents: List of Document objects
        persist_directory: Directory to store the Chroma database
        embedding_model_name: Name of the embedding model to use

    Returns:
        Chroma vector store

    """
    # Create directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Create and persist Chroma database
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)

    vectorstore.persist()
    logger.info(f'Loaded {len(documents)} documents into Chroma at {persist_directory}')

    return vectorstore


def main():
    """Load test data into ChromaDB."""
    import argparse

    parser = argparse.ArgumentParser(description='Load test data into Chroma database')
    parser.add_argument('--output-dir', type=str, default='./chroma_db', help='Directory to store the Chroma database')
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2', help='Embedding model to use')

    args = parser.parse_args()

    try:
        # Create documents
        documents = create_documents(SAMPLE_TEXTS)

        # Split documents
        split_docs = split_documents(documents)

        # Load into Chroma
        vectorstore = load_documents_to_chroma(
            documents=split_docs, persist_directory=args.output_dir, embedding_model_name=args.embedding_model
        )

        # Test search
        results = vectorstore.similarity_search('What is RAG?', k=2)
        logger.info(f'Test search returned {len(results)} results')
        for i, doc in enumerate(results):
            logger.info(f'Result {i + 1}: {doc.metadata.get("title", "Unknown")}')

        print(f'\nSuccessfully loaded {len(split_docs)} document chunks into {args.output_dir}')
        print('You can now run the test_rag_cli.py script to test the RAG system')

    except Exception as e:
        logger.error(f'Error loading test data: {e}', exc_info=True)
        print(f'Error: {e}')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
