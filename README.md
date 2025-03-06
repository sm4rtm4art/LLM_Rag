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

Our CI/CD pipeline automatically uploads coverage reports to Codecov, which provides detailed insights into which parts of the codebase are covered by tests.

### Bash Script Tests

To test the bash scripts in the project, use the following command:

```bash
scripts/test_bash_scripts.sh
```

This will run shellcheck on all bash scripts and perform basic validation. For more thorough testing, install bats-core:

```bash
brew install bats-core  # macOS
```

### RAG Evaluation Tests

To run the RAG evaluation tests specifically:

```bash
scripts/run_rag_tests.sh
```

These tests evaluate the quality of the RAG system's responses against a set of reference answers.

### Docker Tests

To test the Docker build and functionality:

```bash
scripts/test_docker.sh
```

### Kubernetes Tests

To test the Kubernetes configurations:

```bash
scripts/test_kubernetes.sh
```

## Code Quality

This project maintains high code quality standards through:

- **Type Checking**: Mypy ensures type safety throughout the codebase
- **Linting**: Ruff enforces consistent code style and catches potential issues
- **Pre-commit Hooks**: Automated checks run before each commit
- **Continuous Integration**: All PRs are automatically tested
- **Security Scanning**: Bandit, Safety, and Trivy scan for vulnerabilities

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment. The pipeline includes:

### Main CI/CD Workflow

The main workflow (`ci-cd.yml`) runs on pushes to the main branch and pull requests:

1. **Bash Linting**: Checks shell scripts for errors and style issues
2. **Security Scanning**: Runs Bandit, Safety, and Trivy for security vulnerabilities
3. **Python Linting**: Runs Ruff and Mypy for code quality and type checking
4. **Testing**: Runs the test suite with coverage reporting
5. **Semantic Release**: Automatically versions and releases the package
6. **Docker Build**: Builds and pushes the Docker image to Docker Hub
7. **Deployment**: Deploys to development or staging environments

### Kubernetes Testing Workflow

A separate workflow (`k8s-test.yaml`) tests the Kubernetes deployment:

1. Builds or pulls the Docker image
2. Creates a KIND cluster
3. Deploys the application to the cluster
4. Runs tests against the deployed application
5. Cleans up resources

This workflow is configured as non-blocking, so failures won't prevent merges to the main branch.

## Docker

The project includes a multi-stage Dockerfile that creates an optimized image for both development and production use. The multi-stage build approach significantly reduces the final image size by separating the build environment from the runtime environment.

To build the Docker image:

```bash
docker build -t llm-rag .
```

To run the container in CLI mode:

```bash
docker run -it llm-rag
```

To run the container in API mode:

```bash
docker run -it -p 8000:8000 llm-rag api
```

## Kubernetes Deployment

The project includes Kubernetes configuration files for deploying the application to a Kubernetes cluster. See the [k8s/README.md](k8s/README.md) file for detailed instructions.

For local testing with KIND (Kubernetes IN Docker):

```bash
# Create a KIND cluster
kind create cluster --name llm-rag-cluster --config k8s/kind-config.yaml

# Deploy the application
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

## Implementation Details

### Document Processing

The system processes PDF documents in multiple stages:

1. **Text Extraction**: Extract raw text content from PDF pages
2. **Table Detection**: Identify and extract tables using tabula-py
3. **Image Extraction**: Extract images and convert them to a processable format
4. **Chunking**: Split the document into semantic chunks while preserving context

### Vector Store

We use ChromaDB as the vector database, with collections for:

- Text chunks and their embeddings
- Image embeddings and their metadata
- Table content and structure

### LLM Integration

The system currently supports the following LLM backends:

- Hugging Face models (deployed locally or via API)
- Llama.cpp for local deployment of open-source models

Future plans may include integration with additional LLM providers as needed.

### Evaluation Framework

The system includes a comprehensive evaluation framework for assessing RAG performance:

- **Relevance Metrics**: Evaluate the relevance of retrieved documents
- **Answer Quality**: Assess the quality and accuracy of generated responses
- **Context Utilization**: Measure how effectively the system uses retrieved context
- **Automated Testing**: CI/CD integration for continuous evaluation

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python -m pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

We use several tools to maintain code quality:

- **Ruff**: For linting and formatting
- **Mypy**: For type checking
- **Pre-commit**: For automated checks before commits

To set up the development environment:

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
