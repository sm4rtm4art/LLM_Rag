#!/usr/bin/env python
"""Advanced demo script showing how to use DIN-formatted XML documents.

in a RAG (Retrieval-Augmented Generation) system.

This example demonstrates:
1. Loading a DIN document with the XMLLoader
2. Creating vector embeddings for each section
3. Implementing a simple retrieval system
4. Answering questions using the retrieved content

Requirements:
- sentence-transformers (pip install sentence-transformers)
- A DIN XML document (tests/test_data/xml/din_document.xml)
"""

import os
from typing import Dict, List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print('Warning: sentence-transformers not available. Install with: pip install sentence-transformers')

from llm_rag.document_processing.loaders import XMLLoader

# Update path to the DIN document
DIN_DOCUMENT_PATH = 'tests/test_data/xml/din_document.xml'


class SimpleRAGSystem:
    """A simple RAG system using the DIN document and vector embeddings."""

    def __init__(self, xml_file_path: str, use_sections: bool = True):
        """Initialize the RAG system.

        Parameters
        ----------
        xml_file_path : str
            Path to the DIN XML file
        use_sections : bool, optional
            Whether to split the document into sections, by default True

        """
        self.xml_file_path = xml_file_path
        self.use_sections = use_sections
        self.documents = []
        self.embeddings = []
        self.model = None

        # Check if embeddings are available
        if not EMBEDDINGS_AVAILABLE:
            print('Warning: Vector embeddings not available. Running in text-only mode.')

    def load_documents(self):
        """Load documents from the XML file."""
        print(f'Loading documents from {self.xml_file_path}...')

        if self.use_sections:
            # Load each section as a separate document
            loader = XMLLoader(self.xml_file_path, split_by_tag='din:section', metadata_tags=['din:title'])
        else:
            # Load the whole document
            loader = XMLLoader(self.xml_file_path)

        self.documents = loader.load()
        print(f'Loaded {len(self.documents)} documents')

    def create_embeddings(self):
        """Create vector embeddings for all documents."""
        if not EMBEDDINGS_AVAILABLE:
            print('Skipping embeddings creation (dependencies not available)')
            return

        print('Creating embeddings...')

        # Initialize the embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create embeddings for each document
        self.embeddings = []
        for doc in self.documents:
            # Use both the title (if available) and content for better embedding
            text = ''
            if 'din:title' in doc['metadata']:
                text += doc['metadata']['din:title'] + ': '
            text += doc['content']

            # Create and store the embedding
            embedding = self.model.encode(text)
            self.embeddings.append(embedding)

        print(f'Created embeddings for {len(self.embeddings)} documents')

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for relevant documents based on the query.

        Parameters
        ----------
        query : str
            The search query
        top_k : int, optional
            Number of top results to return, by default 3

        Returns
        -------
        List[Tuple[Dict, float]]
            List of (document, score) tuples sorted by relevance

        """
        if EMBEDDINGS_AVAILABLE and self.model:
            return self._vector_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """Perform vector search using embeddings.

        Parameters
        ----------
        query : str
            Search query
        top_k : int
            Number of results to return

        Returns
        -------
        List[Tuple[Dict, float]]
            List of (document, score) tuples

        """
        # Create query embedding
        query_embedding = self.model.encode(query)

        # Calculate cosine similarity with all document embeddings
        scores = []
        for doc_embedding in self.embeddings:
            # Normalize embeddings for cosine similarity
            normalized_query = query_embedding / np.linalg.norm(query_embedding)
            normalized_doc = doc_embedding / np.linalg.norm(doc_embedding)

            # Calculate cosine similarity
            similarity = np.dot(normalized_query, normalized_doc)
            scores.append(similarity)

        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = [(self.documents[i], scores[i]) for i in top_indices]

        return results

    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """Perform simple keyword search as fallback.

        Parameters
        ----------
        query : str
            Search query
        top_k : int
            Number of results to return

        Returns
        -------
        List[Tuple[Dict, float]]
            List of (document, score) tuples

        """
        # Split query into keywords
        keywords = query.lower().split()

        # Calculate scores based on keyword occurrence
        scores = []
        for doc in self.documents:
            content = doc['content'].lower()

            # Count occurrence of each keyword
            score = sum(content.count(keyword) for keyword in keywords)

            # Boost score if keywords appear in title
            if 'din:title' in doc['metadata']:
                title = doc['metadata']['din:title'].lower()
                score += sum(5 * title.count(keyword) for keyword in keywords)

            scores.append(score)

        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = [(self.documents[i], scores[i]) for i in top_indices if scores[i] > 0]

        return results

    def answer_question(self, question: str) -> str:
        """Generate an answer to a question using retrieved documents.

        Parameters
        ----------
        question : str
            The question to answer

        Returns
        -------
        str
            Generated answer with citations

        """
        print(f'\n\nQuestion: {question}')
        print('=' * 80)

        # Search for relevant documents
        relevant_docs = self.search(question, top_k=3)

        if not relevant_docs:
            return "I couldn't find relevant information to answer this question."

        # In a real implementation, you would use an LLM here
        # For this demo, we'll just return the relevant documents
        answer = 'Based on the DIN standard, I found these relevant sections:\n\n'

        for i, (doc, score) in enumerate(relevant_docs, 1):
            title = doc['metadata'].get('din:title', f'Section {i}')
            answer += f'[{i}] {title} (relevance: {score:.2f})\n'

            # Add a snippet from the content
            snippet = doc['content'].strip()
            if len(snippet) > 300:
                snippet = snippet[:300] + '...'
            answer += f'{snippet}\n\n'

        return answer


def run_demo():
    """Run the demonstration."""
    xml_file_path = DIN_DOCUMENT_PATH

    # Make sure the file exists
    if not os.path.exists(xml_file_path):
        print(f'Error: File not found: {xml_file_path}')
        print('Please run this script from the project root directory')
        return

    # Initialize and prepare the RAG system
    rag_system = SimpleRAGSystem(xml_file_path)
    rag_system.load_documents()
    rag_system.create_embeddings()

    # Example questions to demonstrate the system
    questions = [
        'What is RAG and how does it work?',
        'What are the recommended chunking strategies for documents?',
        'How should embedding models be selected for a RAG system?',
        'What metrics should be used to evaluate a RAG system?',
        'What tools and libraries implement the DIN standard for RAG systems?',
    ]

    # Answer each question
    for question in questions:
        answer = rag_system.answer_question(question)
        print(answer)
        print('=' * 80)


if __name__ == '__main__':
    run_demo()
