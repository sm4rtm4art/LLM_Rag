# Tests for the RAG System

This directory contains tests for the RAG system.

## Directory Structure

- `integration/`: Integration tests that test the entire RAG system or multiple components together.
- `unit/`: Unit tests that test individual components of the RAG system.

## Running Tests

You can run the tests using the `scripts/run_tests.py` script:

```bash
# Run all tests
python scripts/run_tests.py --all

# Run unit tests
python scripts/run_tests.py --unit

# Run integration tests
python scripts/run_tests.py --integration

# Run a specific test file
python scripts/run_tests.py --test tests/integration/test_retrieval.py
```

## Writing Tests

When writing tests, follow these guidelines:

1. Place unit tests in the `tests/unit/` directory.
2. Place integration tests in the `tests/integration/` directory.
3. Name test files with the prefix `test_`.
4. Use descriptive names for test functions, prefixed with `test_`.
5. Include docstrings for test functions to describe what they test.
6. Use assertions to verify expected behavior.
