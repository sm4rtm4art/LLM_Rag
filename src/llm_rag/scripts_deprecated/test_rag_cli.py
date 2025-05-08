"""Test script for the RAG CLI interface."""

from src.llm_rag.rag.pipeline.base import RAGPipeline


def main():
    """Test the RAG system with anti-hallucination features."""
    try:
        # Initialize RAG pipeline
        RAGPipeline()
        print('RAG pipeline initialized successfully')
    except Exception as e:
        print(f'Error initializing RAG pipeline: {e}')
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
