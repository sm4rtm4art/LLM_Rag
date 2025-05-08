#!/usr/bin/env python
"""Integration tests for the vector store module."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.documents import Document

from llm_rag.vectorstore.chroma import ChromaVectorStore, EmbeddingFunctionWrapper


class TestChromaVectorStore(unittest.TestCase):
    """Integration tests for the ChromaVectorStore class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the vector store
        self.temp_dir = tempfile.mkdtemp()

        # Create a test embedding model
        self.embeddings = EmbeddingFunctionWrapper(model_name='all-MiniLM-L6-v2')

        # Create a test vector store
        self.vector_store = ChromaVectorStore(
            persist_directory=self.temp_dir, collection_name='test_collection', embedding_function=self.embeddings
        )

        # Create test documents
        self.documents = [
            Document(page_content='This is a test document about AI.', metadata={'source': 'test1.txt', 'page': 1}),
            Document(
                page_content='This document discusses machine learning.', metadata={'source': 'test2.txt', 'page': 1}
            ),
            Document(page_content='Python is a programming language.', metadata={'source': 'test3.txt', 'page': 1}),
        ]

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_add_documents(self):
        """Test adding documents to the vector store."""
        # Act
        self.vector_store.add_documents(self.documents)

        # Assert
        collection_size = self.vector_store.get_collection_size()
        self.assertEqual(collection_size, len(self.documents))

    def test_similarity_search(self):
        """Test similarity search in the vector store."""
        # Arrange
        self.vector_store.add_documents(self.documents)

        # Act
        results = self.vector_store.similarity_search(query='What is artificial intelligence?', k=2)

        # Assert
        self.assertEqual(len(results), 2)
        # The first result should be about AI
        self.assertIn('AI', results[0].page_content)

    def test_persist_and_load(self):
        """Test persisting and loading the vector store."""
        # Arrange
        self.vector_store.add_documents(self.documents)
        self.vector_store.persist()

        # Act
        # Create a new vector store with the same directory
        new_vector_store = ChromaVectorStore(
            persist_directory=self.temp_dir, collection_name='test_collection', embedding_function=self.embeddings
        )

        # Assert
        collection_size = new_vector_store.get_collection_size()
        self.assertEqual(collection_size, len(self.documents))


if __name__ == '__main__':
    unittest.main()
