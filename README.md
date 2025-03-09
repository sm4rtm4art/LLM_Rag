# Multi-Modal RAG System for Standardized Documents

[![CI/CD Pipeline](https://github.com/sm4rtm4art/LLM_Rag/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/sm4rtm4art/LLM_Rag/actions/workflows/ci-cd.yml)
[![Kubernetes Tests](https://github.com/sm4rtm4art/LLM_Rag/actions/workflows/k8s-test.yaml/badge.svg)](https://github.com/sm4rtm4art/LLM_Rag/actions/workflows/k8s-test.yaml)
[![codecov](https://codecov.io/gh/sm4rtm4art/llm-rag/branch/main/graph/badge.svg)](https://codecov.io/gh/sm4rtm4art/llm-rag)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a multi-modal Retrieval-Augmented Generation (RAG) system designed for processing and querying standardized technical documents. The system leverages both text and image content from PDF documents to provide comprehensive responses to user queries.

## Features

- **Multi-Modal Document Processing**: Extract and process both text and tables/images from PDF documents
- **Specialized Chunking**: Intelligent document chunking that preserves context and structure
- **Multi-Modal Vector Store**: Store and retrieve both text and image embeddings
- **Sharded Vector Store**: Horizontal scaling with automatic sharding for improved performance with large document collections
- **Conversational Interface**: Natural language interface for querying document content
- **Flexible LLM Integration**: Support for various LLM backends (Hugging Face, Llama.cpp)
- **Kubernetes Deployment**: Ready-to-use Kubernetes configuration for scalable deployment
- **CI/CD Pipeline**: Comprehensive GitHub Actions workflow for testing, building, and deploying

## Repository Structure

```
.
├── src/                    # Source code for the RAG system
│   └── llm_rag/            # Main package
│       ├── document_processing/  # Document extraction and chunking
│       ├── embeddings/     # Text and image embedding models
│       ├── vectorstore/    # Vector database integration
│       ├── llm/            # LLM integration
│       ├── evaluation/     # RAG evaluation framework
│       └── api/            # FastAPI application
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── evaluation/         # RAG evaluation tests
├── demos/                  # Demo scripts and examples
├── scripts/                # Utility scripts
├── k8s/                    # Kubernetes deployment files
├── data/                   # Sample data and documents
│   └── documents/          # PDF documents for processing
├── notebooks/              # Jupyter notebooks for exploration
├── .github/workflows/      # GitHub Actions CI/CD workflows
└── docs/                   # Documentation
```

## Installation

### Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip

### Using UV (Recommended)

```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/llm-rag.git
cd llm-rag

# Create a virtual environment and install dependencies
uv venv
source .llm_rag/bin/activate  # On Windows: .llm_rag\Scripts\activate
uv pip install -e .

# For development
uv pip install -e ".[dev]"
```

### Using Docker

```bash
# Build the Docker image
docker build -t llm-rag .

# Run the container in CLI mode
docker run -p 8000:8000 llm-rag

# Run the container in API mode
docker run -p 8000:8000 llm-rag api

# Run with specific arguments
docker run llm-rag --help
```

## Usage

### Demo Scripts

The repository includes several demo scripts to showcase different functionalities:

```bash
# Process a PDF document
python -m demos.process_document --pdf_path data/documents/example.pdf

# Query the RAG system
python -m demos.query_rag --query "What are the requirements for steel structures?"

# Run the API server
python -m uvicorn llm_rag.api.main:app --host 0.0.0.0 --port 8000
```

### Command-Line Options

Most scripts support the following options:

- `--pdf_path`: Path to the PDF document
- `--db_path`: Path to the vector database
- `--model_name`: Name of the embedding model to use
- `--llm_provider`: LLM provider to use (huggingface, llamacpp)
- `--verbose`: Enable verbose logging

### API Endpoints

When running in API mode, the following endpoints are available:

- `GET /`: Root endpoint with API information
- `GET /health`: Health check endpoint
- `POST /query`: Process a single query
- `POST /conversation`: Process a query in conversation mode

Example API request:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the requirements for steel structures?", "top_k": 5}'
```

## Testing

### Python Tests

To run the tests, use the following command:

```bash
python -m pytest
```

For more verbose output, use:

```bash
python -m pytest -v
```

For test coverage reporting:

```bash
python -m pytest --cov=src/llm_rag --cov-report=xml --cov-report=term
```
