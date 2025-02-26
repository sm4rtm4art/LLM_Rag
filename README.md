# LLM RAG Project

[![codecov](https://codecov.io/gh/sm4rtm4art/LLM_Rag/branch/main/graph/badge.svg?token=YOUR_CODECOV_TOKEN)](https://codecov.io/gh/sm4rtm4art/LLM_Rag)

## 🚀 Overview

This project implements a Retrieval-Augmented Generation (RAG) system using Large Language Models. It provides a scalable architecture for document processing, embedding generation, and intelligent query answering.

## 🛠 Tech Stack

- Python 3.12
- LangChain for LLM orchestration
- Vector storage with ChromaDB
- FastAPI for API endpoints
- Docker & Kubernetes for deployment
- UV for dependency management
- Comprehensive CI/CD pipeline

## 📋 Prerequisites

- Python 3.12+
- Docker (optional, for containerization)
- Git

## 🔧 Development Setup

### 1. Install UV Package Manager

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

### 2. Clone the Repository

```bash
git clone https://github.com/sm4rtm4art/LLM_Rag.git
cd LLM_Rag
```

### 3. Create and Activate Virtual Environment

```bash
# Create a new virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
# Install project dependencies
uv pip install .

# Install development dependencies
uv pip install ".[dev]"
```

### 5. Set Up Pre-commit Hooks

```bash
# Install pre-commit
uv pip install pre-commit

# Install the git hooks
pre-commit install

# Run against all files (optional)
pre-commit run --all-files
```

## 🧪 Running Tests

```bash
# Run tests with coverage
pytest --cov=src/llm_rag tests/ --cov-report=xml

# Run specific test file
pytest tests/path/to/test_file.py
```

## 🔍 Code Quality

```bash
# Format code
ruff format .

# Run linter
ruff check .

# Run type checker
mypy src tests
```

## 🐳 Docker Support

Build and run the application using Docker:

```bash
# Build the image
docker build -t llm-rag .

# Run the container
docker run -p 8000:8000 llm-rag
```

Using BuildKit for faster builds:

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build with cache
docker buildx build --cache-from=type=registry,ref=theflyingd/llm-rag:buildcache \
                   --cache-to=type=registry,ref=theflyingd/llm-rag:buildcache,mode=max \
                   -t theflyingd/llm-rag:latest .
```

## 📦 Project Structure

llm-rag/
├── src/
│ └── llm_rag/
│ ├── api/ # FastAPI endpoints
│ ├── vectorstore/ # Vector storage implementations
│ └── main.py # Application entry point
├── tests/ # Test suite
├── .github/ # GitHub Actions workflows
├── k8s/ # Kubernetes manifests
├── pyproject.toml # Project metadata and dependencies
└── Dockerfile # Container definition

````

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
# API Keys and Configurations
OPENAI_API_KEY=your_api_key_here
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
````

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
