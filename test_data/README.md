# Test Data Directory

This directory contains sample test documents used for testing the RAG pipeline.

## Purpose

- Provides a minimal set of test documents for unit and integration tests
- Contains public domain content safe for distribution
- Used by automated tests in both local and CI environments

## Contents

- `test.txt`: A simple text file with sample content for testing document loading and processing

## Usage in Tests

This directory is automatically created by the pre-commit hooks when running tests. In CI environments, the directory is mocked to ensure tests pass without requiring actual files.

For local development, real files are used when available, providing more thorough testing of the document processing pipeline.

## Adding Test Files

When adding new test files to this directory, ensure they:

1. Contain only public domain content
2. Are small in size (preferably under 10KB)
3. Are representative of the types of documents your application processes
4. Do not contain sensitive or copyrighted information
