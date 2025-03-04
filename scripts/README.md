# Scripts for the RAG System

This directory contains scripts for the RAG system.

## Directory Structure

- `utils/`: Utility scripts for the RAG system.
- `checks/`: Scripts for checking the RAG system.
- `run_tests.py`: Script for running tests.

## Utility Scripts

The `utils/` directory contains utility scripts for the RAG system:

- `load_documents.py`: Load documents into the RAG system's vector store.
- `check_db_content.py`: Check the content of the vector database.
- `create_test_db.py`: Create a test vector database.
- `inspect_chroma_db.py`: Inspect a Chroma database.
- `download_model.py`: Download a model for the RAG system.

## Check Scripts

The `checks/` directory contains scripts for checking the RAG system:

- `check_vectorstore.py`: Check the vector store.

## Running Tests

You can run the tests using the `run_tests.py` script:

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
