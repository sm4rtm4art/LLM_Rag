# Public Test Data for RAG System

This directory contains public test data that can be safely shared on GitHub and used for testing the RAG system. These files do not contain any sensitive information and are intended for demonstration and testing purposes only.

## Files

- `sample.txt`: A general overview of Retrieval-Augmented Generation (RAG) systems, their benefits, challenges, and applications.
- `llama3_info.txt`: Information about Meta's Llama 3 large language model, including its features, capabilities, and use cases.
- `rag_systems.txt`: Detailed information about RAG systems, their components, and how they work.
- `test_queries.json`: A collection of test queries that can be used to test the RAG system with the public test data.

## Usage

These files can be used with the `test_rag_public.py` script to test the RAG system:

```bash
python scripts/test_rag_public.py --load-documents --collection public_test --query "What is RAG?"
```

Or you can use one of the predefined test queries from `test_queries.json`:

```bash
python scripts/test_rag_public.py --load-documents --collection public_test --use-test-query 0
```

## Adding New Test Data

When adding new test data to this directory, please ensure that:

1. The data does not contain any sensitive or proprietary information
2. The data is relevant for testing the RAG system
3. The data is documented in this README file

## Test Queries

The `test_queries.json` file contains a collection of test queries that can be used to test the RAG system. Each query includes:

- `query`: The question to ask the RAG system
- `expected_answer`: The expected answer from the RAG system
- `expected_sources`: The expected sources that should be retrieved for the query

These test queries can be used for automated testing to ensure that the RAG system is working correctly.
