# Check Scripts for the RAG System

This directory contains scripts for checking the RAG system.

## Scripts

- `check_vectorstore.py`: Check the vector store.

## Usage

### check_vectorstore.py

Check the vector store:

```bash
python scripts/checks/check_vectorstore.py --db-path data/vectorstore --collection-name my_collection
```

## Available Scripts

### `check_import.py`

Checks if required Python modules are available for import. This is useful for verifying that all dependencies are correctly installed.

Usage:

```bash
python scripts/checks/check_import.py
```

### `check_pdf_content.py`

Extracts and checks the content of PDF files in the test data directory. This helps verify that PDF extraction is working correctly.

Usage:

```bash
python scripts/checks/check_pdf_content.py
```

## Running All Checks

To run all checks in sequence, you can use:

```bash
for script in scripts/checks/check_*.py; do python "$script"; done
```

## Adding New Checks

When adding new check scripts, please follow these guidelines:

1. Use the naming convention `check_<feature>.py`
2. Include proper docstrings and error handling
3. Configure logging using the standard format
4. Update this README.md with documentation for the new script
