# Synthetic Test Database

This directory contains a synthetic test database for the LLM RAG system. It is created using public domain text and is safe to commit to the repository.

## Purpose

The synthetic database is used for testing the retrieval functionality of the RAG system without relying on potentially sensitive or copyrighted documents. It contains:

1. General information about RAG systems
2. Information about LLaMA 3 models
3. Other public domain text

## Usage

This database is automatically used by the test suite. If you want to use it in your own tests, you can specify it as the `persist_directory` when creating a `ChromaVectorStore`:

```python
from src.llm_rag.vectorstore.chroma import ChromaVectorStore

vectorstore = ChromaVectorStore(
    collection_name="test_collection",
    persist_directory="synthetic_test_db"
)
```

## Regenerating the Database

If you need to regenerate the database, you can run the `create_test_db.py` script:

```bash
python create_test_db.py
```

This will create a new synthetic database with fresh public domain text.
