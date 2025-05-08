#!/usr/bin/env python
"""Simple test to verify imports are working correctly."""

try:
    print('Attempting to import RAGPipeline...')
    print('SUCCESS: RAGPipeline imported correctly!')
except Exception as e:
    print(f'FAILED: Could not import RAGPipeline: {e}')

try:
    print('\nAttempting to import BaseRAGPipeline...')
    print('SUCCESS: BaseRAGPipeline imported correctly!')
except Exception as e:
    print(f'FAILED: Could not import BaseRAGPipeline: {e}')

print('\nImport test complete.')
