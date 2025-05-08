#!/usr/bin/env python
"""Test LlamaIndex imports to diagnose what's actually available."""

import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_import(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        logger.info(f'✅ Successfully imported {module_name}')
        return True
    except ImportError as e:
        logger.error(f'❌ Failed to import {module_name}: {e}')
        return False


def main():
    """Test various LlamaIndex imports."""
    # Check if llama_index is installed
    if not check_import('llama_index'):
        logger.error('llama_index not installed. Try: pip install llama-index')
        return

    # Basic imports - these should always work if llama_index is installed
    modules_to_test = [
        'llama_index',
        'llama_index.core',
        'llama_index.core.schema',
        'llama_index.core.node_parser',
        'llama_index.embeddings',
        'llama_index.llms',
        'llama_index.vector_stores',
    ]

    for module in modules_to_test:
        check_import(module)

    # Check specific components
    try:
        import llama_index

        logger.info(f'LlamaIndex version: {llama_index.__version__}')

        # Print available submodules
        logger.info('Available submodules in llama_index:')
        for attr in dir(llama_index):
            if not attr.startswith('_'):
                logger.info(f'  - {attr}')

    except ImportError:
        pass

    # Try to import and print the correct import path for ChromaVectorStore
    try:
        # Different possible locations to check
        possible_paths = [
            'from llama_index.vector_stores.chroma import ChromaVectorStore',
            'from llama_index_vector_stores_chroma import ChromaVectorStore',
            'from llama_index.core.vector_stores import ChromaVectorStore',
        ]

        for import_statement in possible_paths:
            try:
                logger.info(f'Trying: {import_statement}')
                exec(import_statement)
                logger.info(f'✅ Success: {import_statement}')
                break
            except ImportError as e:
                logger.error(f'❌ Failed: {import_statement} - {e}')

    except Exception as e:
        logger.error(f'Error during import testing: {e}')


if __name__ == '__main__':
    main()
