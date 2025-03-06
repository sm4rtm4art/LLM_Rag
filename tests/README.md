# Tests for the RAG System

[![codecov](https://codecov.io/gh/sm4rtm4art/llm-rag/branch/main/graph/badge.svg)](https://codecov.io/gh/sm4rtm4art/llm-rag)

This directory contains tests for the RAG system.

## Directory Structure

- `integration/`: Integration tests that test the entire RAG system or multiple components together.
- `unit/`: Unit tests that test individual components of the RAG system.
- `evaluation/`: Tests that evaluate the quality of the RAG system's responses.
- `promptfoo/`: Tests using promptfoo for LLM response evaluation.
- `test_data/`: Test data used by the tests.

## Running Tests

You can run the tests using pytest directly:

```bash
# Run all tests
python -m pytest

# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/

# Run a specific test file
python -m pytest tests/integration/test_retrieval.py

# Run tests with coverage
python -m pytest --cov=src/llm_rag --cov-report=xml --cov-report=term
```

Or you can use the `scripts/run_tests.py` script:

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

## Test Coverage

We use pytest-cov to measure test coverage. The CI/CD pipeline automatically uploads coverage reports to Codecov, which provides detailed insights into which parts of the codebase are covered by tests.

To view the coverage report locally:

```bash
python -m pytest --cov=src/llm_rag --cov-report=html
# Then open htmlcov/index.html in your browser
```

## CI/CD Integration

Tests are automatically run in our CI/CD pipeline on every push and pull request. The pipeline:

1. Runs all tests with coverage reporting
2. Uploads coverage reports to Codecov
3. Fails the build if tests fail

## Writing Tests

When writing tests, follow these guidelines:

1. Place unit tests in the `tests/unit/` directory.
2. Place integration tests in the `tests/integration/` directory.
3. Name test files with the prefix `test_`.
4. Use descriptive names for test functions, prefixed with `test_`.
5. Include docstrings for test functions to describe what they test.
6. Use assertions to verify expected behavior.
7. Aim for high test coverage, especially for critical components.
8. Use fixtures and mocks to isolate tests and make them faster.

## Test Data

Test data is stored in the `tests/test_data/` directory. This includes:

- Sample PDF documents
- Expected extraction results
- Query-answer pairs for evaluation

When adding new test data, make sure to document its purpose and structure.
