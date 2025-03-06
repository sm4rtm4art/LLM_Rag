# RAG System Testing Approach

[![CI/CD Pipeline](https://github.com/sm4rtm4art/LLM_Rag/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/sm4rtm4art/LLM_Rag/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/sm4rtm4art/llm-rag/branch/main/graph/badge.svg)](https://codecov.io/gh/sm4rtm4art/llm-rag)

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
- CI/CD integration: Tests are automatically run in the CI/CD pipeline

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

### With Coverage Reporting

```bash
pytest tests/test_rag_evaluation.py --cov=src/llm_rag/rag --cov-report=term
```

## CI/CD Integration

The RAG evaluation tests are fully integrated into our CI/CD pipeline:

1. **Automated Testing**: Tests run automatically on every push and pull request
2. **Coverage Reporting**: Test coverage is reported to Codecov
3. **Pre-merge Checks**: Tests must pass before merging to main branch
4. **Non-blocking Tests**: Some advanced RAG tests are configured as non-blocking to prevent blocking the pipeline for subjective quality assessments

### GitHub Actions Configuration

The tests are configured in the `.github/workflows/ci-cd.yml` file:

```yaml
- name: Run RAG Evaluation Tests
  run: |
    python -m pytest tests/test_rag_evaluation.py -v
```

## Extending Tests

### Adding New Test Cases

1. Add new queries to `tests/test_data/test_queries.json` with expected answers and sources
2. The test framework will automatically pick up the new test cases

### Adding New Test Types

1. Add new test functions to `tests/test_rag_evaluation.py`
2. Follow the existing patterns for mocking and assertions

### Adding New Metrics

1. Implement new evaluation metrics in `src/llm_rag/evaluation/metrics.py`
2. Add tests for the new metrics in `tests/test_rag_evaluation.py`

## Benefits of This Approach

1. **Integration with Existing Tools**: Leverages the existing pytest framework
2. **Developer Familiarity**: Uses tools developers are already familiar with
3. **Maintainability**: Easier to maintain as part of the regular codebase
4. **Automation**: Integrated with pre-commit hooks for automatic testing
5. **Extensibility**: Easy to extend with additional test cases and metrics
6. **CI/CD Integration**: Seamlessly integrated with the CI/CD pipeline
7. **Coverage Tracking**: Test coverage is tracked and reported

## Future Improvements

1. **More Sophisticated Metrics**: Implement more advanced similarity metrics
2. **Performance Testing**: Add tests for response time and resource usage
3. **Edge Case Testing**: Add more tests for edge cases and error handling
4. **Regression Testing**: Implement regression tests to catch regressions in model performance
5. **Automated Benchmark**: Create an automated benchmark for comparing different RAG configurations
6. **A/B Testing**: Implement A/B testing for comparing different RAG approaches
7. **Integration with Model Monitoring**: Connect with model monitoring tools for production deployments

## Troubleshooting

If you encounter issues with the RAG evaluation tests:

1. **Test Data Issues**: Verify that the test data in `tests/test_data/test_queries.json` is correctly formatted
2. **Model Access Issues**: Ensure that the test environment has access to the required models
3. **Memory Issues**: For large models, you may need to increase the available memory
4. **Timeout Issues**: For slow models, you may need to increase the test timeout

## Related Documentation

- [Main README](../README.md): Overview of the entire project
- [Tests README](README.md): General testing approach
- [Evaluation Module Documentation](../src/llm_rag/evaluation/README.md): Details about the evaluation metrics
