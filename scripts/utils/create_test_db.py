#!/usr/bin/env python

"""Create a test vector database.

This script creates a test vector database with sample documents for testing
the RAG system.
"""

import logging
import os
import shutil
import sys
from typing import Any, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Add the project root to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Sample public domain texts (from Project Gutenberg or similar sources)
SAMPLE_TEXTS = [
    "Artificial intelligence (AI) is intelligence demonstrated by machines, "
    "unlike the natural intelligence displayed by humans and animals.",
    "Machine learning is the study of computer algorithms that improve automatically through experience.",
    "Natural Language Processing (NLP) is a field of AI that gives machines "
    "the ability to read, understand, and derive meaning from human languages.",
    "Computer vision is a field of AI that enables computers to derive "
    "meaningful information from digital images, videos, and other visual "
    "inputs.",
    "Reinforcement learning is an area of machine learning concerned with how "
    "software agents ought to take actions in an environment to maximize some "
    "notion of cumulative reward.",
    "Deep learning is a subset of machine learning that uses neural networks "
    "with multiple layers to analyze various factors of data.",
    "Supervised learning is the machine learning task of learning a function "
    "that maps an input to an output based on example input-output pairs.",
    "Unsupervised learning is a type of machine learning algorithm used to draw "
    "inferences from datasets consisting of input data without labeled "
    "responses.",
    "Transfer learning is a research problem in machine learning that focuses "
    "on storing knowledge gained while solving one problem and applying it to "
    "a different but related problem.",
    "Generative AI refers to artificial intelligence systems that can generate "
    "new content, such as text, images, audio, or video, based on patterns "
    "learned from existing data.",
    "Large Language Models (LLMs) are deep learning models trained on vast "
    "amounts of text data that can generate human-like text and perform a "
    "variety of language tasks.",
    "Retrieval-Augmented Generation (RAG) combines retrieval systems with "
    "generative models to generate more accurate and contextual responses.",
    "Vector databases store embeddings, which are numerical representations of "
    "data that capture semantic meaning, allowing for similarity search and "
    "efficient retrieval of related information.",
    "Embeddings are dense vector representations of data (text, images, etc.) "
    "that capture semantic meaning and enable machines to understand "
    "relationships between different pieces of content.",
]

# Categories for metadata
CATEGORIES = [
    "AI Basics",
    "Machine Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Deep Learning",
    "Generative AI",
    "Vector Databases",
]


def generate_metadata(index: int) -> Dict[str, Any]:
    """Generate metadata for a document.

    Args:
        index: The index of the document.

    Returns:
        A dictionary containing metadata for the document.

    """
    return {
        "id": str(index),
        "category": CATEGORIES[index % len(CATEGORIES)],
        "importance": (index % 5) + 1,  # 1-5 scale
        "source": "test_data",
    }


def create_synthetic_db(persist_directory: str = "test_chroma_db", collection_name: str = "test_collection") -> None:
    """Create a synthetic vector database for testing.

    Args:
        persist_directory: The directory to persist the database to.
        collection_name: The name of the collection to create.

    """
    # Create the directory if it doesn't exist
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

    # Initialize the embedding function
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Create the Chroma database
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )

    # Add documents to the database
    metadatas = [generate_metadata(i) for i in range(len(SAMPLE_TEXTS))]
    db.add_texts(texts=SAMPLE_TEXTS, metadatas=metadatas)

    # Persist the database
    db.persist()

    logging.info(f"Created synthetic database with {len(SAMPLE_TEXTS)} documents")
    logging.info(f"Database persisted to {persist_directory}")

    # Create a README file explaining the database
    readme_path = os.path.join(persist_directory, "README.md")
    with open(readme_path, "w") as f:
        f.write("""# Test Database

This directory contains a test database for the LLM RAG system. It is created using
synthetic data for testing purposes.

## Purpose

The test database is used for testing the retrieval functionality of the RAG system
without relying on potentially sensitive or copyrighted documents.

## Content

The database contains sample texts about:
- Artificial Intelligence concepts
- Machine Learning
- Natural Language Processing
- Computer Vision
- And other AI-related topics

## Usage

This database is automatically created by the `create_test_db.py` script and is used
by the test suite. You can also use it for manual testing.

## Note

This database contains only public, synthetic data and is safe to include in the
repository.
""")


if __name__ == "__main__":
    create_synthetic_db()
    logging.info("Test database created successfully")
