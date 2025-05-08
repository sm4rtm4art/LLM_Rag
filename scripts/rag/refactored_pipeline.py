"""Basic test for the refactored RAG pipeline.

This script tests the basic functionality of the refactored RAG pipeline
to ensure that it's working as expected.
"""

import sys
from unittest.mock import MagicMock

# Add the src directory to the path so we can import from llm_rag
sys.path.insert(0, '../../..')

from src.llm_rag.rag.pipeline import (
    ConversationalRAGPipeline,
    RAGPipeline,
    create_formatter,
    create_generator,
    create_retriever,
)


def test_basic_components():
    """Test that we can create and use the basic components."""
    # Create mock objects
    mock_vectorstore = MagicMock()
    mock_llm = MagicMock()

    # Configure mock behavior
    mock_documents = [
        {'content': 'Test document 1', 'metadata': {'source': 'test1.txt'}},
        {'content': 'Test document 2', 'metadata': {'source': 'test2.txt'}},
    ]
    mock_vectorstore.similarity_search.return_value = mock_documents
    mock_llm.invoke.return_value.content = 'This is a test response.'

    # Create components
    retriever = create_retriever(source=mock_vectorstore, top_k=2)
    formatter = create_formatter(format_type='simple', include_metadata=True)
    generator = create_generator(llm=mock_llm, apply_anti_hallucination=False)

    # Test retriever
    print('Testing retriever...')
    documents = retriever.retrieve('test query')
    assert len(documents) == 2, f'Expected 2 documents, got {len(documents)}'
    print('✅ Retriever works')

    # Test formatter
    print('Testing formatter...')
    context = formatter.format_context(documents)
    assert isinstance(context, str), f'Expected string context, got {type(context)}'
    assert 'Test document 1' in context, 'Expected document content in context'
    print('✅ Formatter works')

    # Test generator
    print('Testing generator...')
    response = generator.generate(query='test query', context=context)
    assert isinstance(response, str), f'Expected string response, got {type(response)}'
    assert response == 'This is a test response.', f'Unexpected response: {response}'
    print('✅ Generator works')

    print('All component tests passed!')


def test_create_pipeline():
    """Test that we can create a RAG pipeline from the components."""
    # Create mock objects
    mock_vectorstore = MagicMock()
    mock_llm = MagicMock()

    # Create pipeline
    pipeline = RAGPipeline(
        vectorstore=mock_vectorstore,
        llm=mock_llm,
        top_k=2,
    )

    assert pipeline is not None, 'Failed to create pipeline'
    print('✅ RAGPipeline created successfully')

    # Create conversational pipeline
    conv_pipeline = ConversationalRAGPipeline(
        vectorstore=mock_vectorstore,
        llm=mock_llm,
        top_k=2,
    )

    assert conv_pipeline is not None, 'Failed to create conversational pipeline'
    print('✅ ConversationalRAGPipeline created successfully')


if __name__ == '__main__':
    test_basic_components()
    test_create_pipeline()
    print('\n✅ All tests passed!')
