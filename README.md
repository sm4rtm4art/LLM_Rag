# Multi-Modal RAG System for DIN Standards

This repository contains a multi-modal Retrieval-Augmented Generation (RAG) system designed for processing and querying Normed standards documents. The system leverages both text and image content from PDF documents to provide comprehensive responses to user queries.

## Features

- **Multi-Modal Document Processing**: Extract and process both text and tables/images from PDF documents
- **Specialized Chunking**: Intelligent document chunking that preserves context and structure
- **Multi-Modal Vector Store**: Store and retrieve both text and image embeddings
- **Conversational Interface**: Natural language interface for querying document content
- **Flexible LLM Integration**: Support for various LLM backends (OpenAI, Hugging Face, Llama.cpp)
- **Kubernetes Deployment**: Ready-to-use Kubernetes configuration for scalable deployment

## Repository Structure

```
.
├── src/                    # Source code for the RAG system
│   └── llm_rag/            # Main package
│       ├── document_processing/  # Document extraction and chunking
│       ├── embeddings/     # Text and image embedding models
│       ├── vectorstore/    # Vector database integration
│       ├── llm/            # LLM integration
│       └── api/            # FastAPI application
├── tests/                  # Test suite
├── demos/                  # Demo scripts and examples
├── scripts/                # Utility scripts
├── k8s/                    # Kubernetes deployment files
├── data/                   # Sample data and documents
│   └── documents/          # PDF documents for processing
├── notebooks/              # Jupyter notebooks for exploration
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
- `--llm_provider`: LLM provider to use (openai, huggingface, llamacpp)
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

The system supports multiple LLM backends:

- OpenAI API (GPT-3.5, GPT-4) (NOT YET IMPLEMENTED! )
- Hugging Face models (deployed locally or via API)
- Llama.cpp for local deployment of open-source models

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
