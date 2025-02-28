# LLM RAG System

A Python package implementing a Retrieval-Augmented Generation (RAG) system using large language models and vector databases.

## Features

- **Document Processing:** Load and process documents from various sources (text files, CSV, directories)
- **Text Chunking:** Split documents into manageable chunks with customizable strategies
- **Embedding Generation:** Generate embeddings for text using sentence-transformers
- **Vector Storage:** Store and retrieve document embeddings using ChromaDB
- **RAG Pipeline:** Complete pipeline for retrieval-augmented generation
- **Conversational Mode:** Support for maintaining conversation history in RAG interactions
- **API Interface:** FastAPI-based REST API for easy integration

## Installation

### Prerequisites

- Python 3.12+
- pip or uv (package installer)

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-rag.git
cd llm-rag

# Install the package and dependencies
pip install -e .

# For development dependencies
pip install -e '.[dev]'
```

### Environment Variables

For using OpenAI models, you need to set your API key:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Command Line Interface

The package provides a command-line interface for ingesting documents and querying the RAG system:

```bash
# Ingest documents from a directory
python -m llm_rag.main --data-dir /path/to/documents --db-dir ./chroma_db

# Query the system
python -m llm_rag.main --query "Your question about the documents?" --db-dir ./chroma_db

# Start interactive mode
python -m llm_rag.main --db-dir ./chroma_db

# Use a specific embedding model
python -m llm_rag.main --embedding-model all-MiniLM-L6-v2 --db-dir ./chroma_db

# Use a specific OpenAI model
python -m llm_rag.main --model gpt-3.5-turbo-instruct --db-dir ./chroma_db
```

### Python API

You can also use the package as a Python library:

```python
from llm_rag.models.embeddings import EmbeddingModel
from llm_rag.document_processing.loaders import DirectoryLoader
from llm_rag.document_processing.chunking import RecursiveTextChunker
from llm_rag.vectorstore.chroma import ChromaVectorStore
from llm_rag.rag.pipeline import RAGPipeline
from langchain.llms import OpenAI
import chromadb

# Initialize embedding model
embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2")

# Load and process documents
loader = DirectoryLoader("/path/to/documents", recursive=True)
documents = loader.load()

chunker = RecursiveTextChunker(chunk_size=1000, chunk_overlap=200)
chunked_docs = chunker.split_documents(documents)

# Set up vector store
chroma_client = chromadb.PersistentClient(path="./chroma_db")
vector_store = ChromaVectorStore(
    client=chroma_client,
    collection_name="rag_documents",
    embedding_function=embedding_model,
)

# Add documents to vector store
vector_store.add_documents(chunked_docs)

# Initialize language model and RAG pipeline
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
rag_pipeline = RAGPipeline(
    vectorstore=vector_store,
    llm=llm,
    top_k=3,
)

# Query the RAG system
result = rag_pipeline.query("Your question about the documents?")
print(result["response"])
```

### REST API

The package includes a FastAPI-based REST API:

```bash
# Start the API server
uvicorn llm_rag.api.main:app --reload
```

Then you can interact with the API:

```bash
# Example query using curl
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question about the documents?"}'
```

Or visit `http://localhost:8000/docs` in your browser to use the Swagger UI.

## Architecture

The system is built with a modular architecture:

1. **Document Processing**: Handles loading and parsing documents from various sources
2. **Text Chunking**: Splits documents into manageable pieces for embedding
3. **Embedding Generation**: Creates vector representations of text
4. **Vector Storage**: Stores and retrieves document embeddings efficiently
5. **RAG Pipeline**: Orchestrates the retrieval and generation process
6. **API Interface**: Provides REST endpoints for interacting with the system

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/llm_rag

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Code Quality

The project uses several tools to maintain code quality:

- **ruff**: For linting and code style
- **mypy**: For type checking
- **pre-commit**: For automated checks before commits

```bash
# Run linting
ruff check .

# Run type checking
mypy .

# Install pre-commit hooks
pre-commit install
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
