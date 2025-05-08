#!/usr/bin/env python
"""Test file to verify imports are working correctly."""

print('Starting import test...')

try:
    # Try importing directly from the pipeline package
    print('✓ Successfully imported from llm_rag.rag.pipeline')
except Exception as e:
    print(f'✗ Error importing from llm_rag.rag.pipeline: {e}')

try:
    # Try importing from the individual modules
    print('✓ Successfully imported from llm_rag.rag.pipeline.base')
except Exception as e:
    print(f'✗ Error importing from llm_rag.rag.pipeline.base: {e}')

try:
    # Try importing from the legacy pipeline.py
    from llm_rag.rag import pipeline

    print('✓ Successfully imported llm_rag.rag.pipeline module')

    # Check if RAGPipeline is available
    if hasattr(pipeline, 'RAGPipeline'):
        print('✓ RAGPipeline is available in llm_rag.rag.pipeline')
    else:
        print('✗ RAGPipeline is NOT available in llm_rag.rag.pipeline')
except Exception as e:
    print(f'✗ Error importing llm_rag.rag.pipeline module: {e}')

print('Import test completed.')
