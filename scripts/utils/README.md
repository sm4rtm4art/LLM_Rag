# Utility Scripts for the RAG System

This directory contains utility scripts for the RAG system.

## Scripts

- `load_documents.py`: Load documents into the RAG system's vector store.
- `check_db_content.py`: Check the content of the vector database.
- `create_test_db.py`: Create a test vector database.
- `inspect_chroma_db.py`: Inspect a Chroma database.
- `download_model.py`: Download a model for the RAG system.

## Usage

### load_documents.py

Load documents into the RAG system's vector store:

```bash
python scripts/utils/load_documents.py --dir data/documents --glob "*.pdf" --db-path data/vectorstore --collection-name my_collection
```

### check_db_content.py

Check the content of the vector database:

```bash
python scripts/utils/check_db_content.py --db-path data/vectorstore --collection-name my_collection
```

### create_test_db.py

Create a test vector database:

```bash
python scripts/utils/create_test_db.py --output-dir data/test_vectorstore --collection-name test_collection
```

### inspect_chroma_db.py

Inspect a Chroma database:

```bash
python scripts/utils/inspect_chroma_db.py --db-path data/vectorstore
```

### download_model.py

Download a model for the RAG system:

```bash
python scripts/utils/download_model.py --model-name TheBloke/Llama-2-7B-Chat-GGUF --output-dir models
```
