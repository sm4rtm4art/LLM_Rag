# Integration Tests for the RAG System

This directory contains integration tests for the RAG system.

## Tests

- `test_retrieval.py`: Test the retrieval capabilities of the RAG system.
- `test_rag.py`: Test the RAG pipeline with a real LLM.
- `test_rag_query.py`: Test the RAG query functionality with a real LLM.
- `test_llama_cpp.py`: Test the llama.cpp integration with the RAG system.

## Running Tests

You can run the integration tests using the `scripts/run_tests.py` script:

```bash
# Run all integration tests
python scripts/run_tests.py --integration

# Run a specific integration test
python scripts/run_tests.py --test tests/integration/test_retrieval.py
```

## Writing Integration Tests

When writing integration tests, follow these guidelines:

1. Place integration tests in this directory.
2. Name test files with the prefix `test_`.
3. Use descriptive names for test functions, prefixed with `test_`.
4. Include docstrings for test functions to describe what they test.
5. Use assertions to verify expected behavior.
6. Test the interaction between multiple components of the RAG system.
