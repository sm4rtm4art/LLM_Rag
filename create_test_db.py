#!/usr/bin/env python3

"""Script to create a small, synthetic test database for testing purposes.

This creates a database with public domain text that can be safely committed
to a repository.
"""

import logging
import os
import shutil
from typing import Any, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Sample public domain texts (from Project Gutenberg or similar sources)
SAMPLE_TEXTS = [
    "Artificial intelligence (AI) is intelligence demonstrated by machines, "
    "as opposed to intelligence displayed by humans or animals.",
    "Machine learning is a subset of AI that focuses on the development of "
    "algorithms that can learn from and make predictions based on data.",
    "Natural Language Processing (NLP) is a field of AI that gives machines "
    "the ability to read, understand, and derive meaning from human languages.",
    "Computer vision is a field of AI that enables computers to derive "
    "meaningful information from digital images, videos, and other visual inputs.",
    "Reinforcement learning is an area of machine learning concerned with how "
    "software agents ought to take actions in an environment to maximize some "
    "notion of cumulative reward.",
    "Deep learning is a subset of machine learning that uses neural networks "
    "with many layers (hence 'deep') to analyze various factors of data.",
    "A neural network is a series of algorithms that endeavors to recognize "
    "underlying relationships in a set of data through a process that mimics "
    "the way the human brain operates.",
    "Supervised learning is the machine learning task of learning a function "
    "that maps an input to an output based on example input-output pairs.",
    "Unsupervised learning is a type of machine learning algorithm used to "
    "draw inferences from datasets consisting of input data without labeled "
    "responses.",
    "Transfer learning is a research problem in machine learning that focuses "
    "on storing knowledge gained while solving one problem and applying it to "
    "a different but related problem.",
    "RAG (Retrieval-Augmented Generation) is a technique that enhances large "
    "language models by retrieving relevant information from external sources "
    "to generate more accurate and contextual responses.",
    "Vector databases store embeddings, which are numerical representations of "
    "data that capture semantic meaning, allowing for similarity search and "
    "efficient retrieval of related information.",
    "Embeddings are dense vector representations of data (text, images, etc.) "
    "that capture semantic relationships, allowing machines to understand "
    "similarity between different pieces of content.",
    "Prompt engineering is the process of designing and optimizing input "
    "prompts for language models to elicit desired outputs, improving the "
    "quality and relevance of generated content.",
    "Fine-tuning is the process of taking a pre-trained model and further "
    "training it on a specific dataset to adapt it for particular tasks or "
    "domains, improving its performance for those specific applications.",
]


def generate_metadata(index: int) -> Dict[str, Any]:
    """Generate metadata for a sample text.

    Args:
        index: Index of the sample text

    Returns:
        Dictionary containing metadata

    """
    return {
        "source": f"synthetic_document_{index}.txt",
        "filetype": "txt",
        "page_num": 1,
        "section_title": "AI Concepts",
    }


def create_synthetic_db(persist_directory: str = "test_chroma_db", collection_name: str = "test_collection") -> None:
    """Create a synthetic database with sample texts.

    Args:
        persist_directory: Directory to store the database
        collection_name: Name of the collection to create

    Returns:
        None

    """
    # Remove existing database if it exists
    if os.path.exists(persist_directory):
        logger.info(f"Removing existing database at {persist_directory}")
        shutil.rmtree(persist_directory)

    # Create embeddings model
    logger.info("Creating embeddings model")
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Create documents with metadata
    logger.info("Creating documents")
    documents = []
    metadatas = []
    ids = []

    for i, text in enumerate(SAMPLE_TEXTS):
        documents.append(text)
        metadatas.append(generate_metadata(i))
        ids.append(f"doc_{i}")

    # Create vector store
    logger.info(f"Creating vector store at {persist_directory}")
    vectorstore = Chroma.from_texts(
        texts=documents,
        embedding=embedding_function,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    # Persist the database
    vectorstore.persist()

    logger.info(f"Created synthetic database with {len(documents)} documents at {persist_directory}")

    # Create a README file explaining the database
    readme_path = os.path.join(persist_directory, "README.md")
    with open(readme_path, "w") as f:
        f.write("""# Test Database

This directory contains a test database for the LLM RAG system. It is created using
public domain text about AI concepts and is safe for distribution.

## Purpose

The test database is used for testing the retrieval functionality of the RAG system
without relying on potentially sensitive or copyrighted documents.

## Content

The database contains sample texts about:
- Artificial Intelligence concepts
- Machine Learning
- Natural Language Processing
- RAG systems
- And other related topics

All content is public domain and safe for distribution.
""")

    logger.info(f"Created README at {readme_path}")


if __name__ == "__main__":
    create_synthetic_db()
    logger.info("Database creation complete")
