# Check Scripts

This directory contains utility scripts for checking and validating various aspects of the LLM RAG system.

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

### `check_vectorstore.py`

Inspects the contents of the vector store, retrieves documents, and performs similarity searches for specific terms. This is useful for verifying that documents are correctly indexed and can be retrieved.

Usage:

```bash
python scripts/checks/check_vectorstore.py
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
