# RAG System Testing Approach

## Overview

This document outlines the approach for testing the RAG (Retrieval-Augmented Generation) system using pytest. Instead of using external tools like promptfoo, we've integrated testing directly into the existing pytest framework for better maintainability and integration with the development workflow.

## Testing Components

### 1. Test Data

- Located in `tests/test_data/test_queries.json`
- Contains sample queries about DIN standards with expected answers and sources
- Easily extensible for additional test cases

### 2. Test Implementation

- Located in `tests/test_rag_evaluation.py`
- Uses pytest fixtures and mocks to isolate the RAG pipeline from external dependencies
- Tests various aspects of the RAG system:
  - Response quality and relevance
  - Source document inclusion
  - Handling of uncertainty
  - Response consistency

### 3. Evaluation Metrics

- **Jaccard Similarity**: Measures the similarity between actual and expected responses
- **Source Inclusion**: Verifies that the expected source documents are included in the response
- **Content Checks**: Ensures responses contain relevant information and have sufficient length

### 4. Integration with Development Workflow

- Pre-commit hook: Automatically runs RAG evaluation tests before each commit
- Test script: `scripts/run_rag_tests.sh` provides a convenient way to run tests manually
- CI/CD integration: Tests can be easily integrated into continuous integration pipelines

## Running Tests

### Using the Script

```bash
./scripts/run_rag_tests.sh
```

### Using pytest Directly

```bash
pytest tests/test_rag_evaluation.py -v
```

### Using Pre-commit

```bash
pre-commit run pytest-rag-evaluation
```

## Extending Tests

### Adding New Test Cases

1. Add new queries to `tests/test_data/test_queries.json` with expected answers and sources
2. The test framework will automatically pick up the new test cases

### Adding New Test Types

1. Add new test functions to `tests/test_rag_evaluation.py`
2. Follow the existing patterns for mocking and assertions

## Benefits of This Approach

1. **Integration with Existing Tools**: Leverages the existing pytest framework
2. **Developer Familiarity**: Uses tools developers are already familiar with
3. **Maintainability**: Easier to maintain as part of the regular codebase
4. **Automation**: Integrated with pre-commit hooks for automatic testing
5. **Extensibility**: Easy to extend with additional test cases and metrics

## Future Improvements

1. **More Sophisticated Metrics**: Implement more advanced similarity metrics
2. **Performance Testing**: Add tests for response time and resource usage
3. **Edge Case Testing**: Add more tests for edge cases and error handling
4. **Regression Testing**: Implement regression tests to catch regressions in model performance
