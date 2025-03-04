# Multi-Modal RAG System for DIN Standards

This project implements a Retrieval-Augmented Generation (RAG) system specialized for processing DIN standards with multi-modal content, including text, tables, and images/technical drawings.

## Overview

The system is designed to extract, process, and retrieve information from DIN standards documents, providing comprehensive answers to user queries by leveraging different types of content:

- **Text content**: Standard paragraphs, sections, and textual information
- **Table content**: Structured data presented in tables
- **Image content**: Figures, diagrams, and technical drawings

## Features

- **Multi-modal document processing**: Extract and process text, tables, and images from DIN standards
- **Specialized chunking**: Content-aware chunking that preserves the integrity of tables and images
- **Multi-modal vector store**: Separate embedding spaces for different content types
- **Content-aware retrieval**: Retrieve the most relevant content based on the query type
- **Conversational interface**: Interactive query system with memory for follow-up questions

## Components

The system consists of several key components:

1. **Document Processing**

   - `DINStandardLoader`: Specialized loader for DIN standard PDFs
   - `MultiModalChunker`: Content-aware chunker for text, tables, and images

2. **Vector Store**

   - `MultiModalVectorStore`: Vector store with specialized embedding models for different content types
   - `MultiModalEmbeddingFunction`: Custom embedding function for multi-modal content

3. **RAG Pipeline**
   - `MultiModalRAGPipeline`: Specialized RAG pipeline for multi-modal content
   - Custom prompt templates for comprehensive answers

## Installation

This project uses [UV](https://github.com/astral-sh/uv) for package management instead of pip. UV is a fast, reliable Python package installer and resolver.

```bash
# Clone the repository
git clone https://github.com/yourusername/din-multimodal-rag.git
cd din-multimodal-rag

# Install UV if you don't have it already
curl -sSf https://astral.sh/uv/install.sh | bash

# Create a virtual environment
python -m venv .llm_rag
source .llm_rag/bin/activate  # On Windows: .llm_rag\Scripts\activate

# Install dependencies using UV
uv pip install -e .
```

### Development Setup

For development, install the project with development dependencies:

```bash
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Usage

### Demo Script

The project includes a demo script (`demo_din_multimodal_rag.py`) that showcases the multi-modal RAG system:

```bash
python demo_din_multimodal_rag.py --din_path /path/to/din/standards
```

### Command-line Options

- `--din_path`: Path to DIN standard PDF or directory containing DIN standards (required)
- `--model`: Name or path of the model to use (default: "microsoft/phi-2")
- `--device`: Device to use for model inference ("cpu", "cuda", "auto") (default: "auto")
- `--persist_dir`: Directory to persist the vector store (default: "chroma_din_multimodal")
- `--top_k`: Number of documents to retrieve per content type (default: 3)
- `--no_tables`: Disable table extraction
- `--no_images`: Disable image extraction
- `--no_drawings`: Disable technical drawing identification

### Interactive Query Session

The demo script provides an interactive query session where you can ask questions about the DIN standards:

```
=== DIN Standards Multi-Modal RAG Demo ===
Type 'exit' or 'quit' to end the session
Type 'reset' to reset the conversation history

Enter your query: What are the safety requirements for machine tools?

=== Response ===
[Detailed response about safety requirements in DIN standards]

=== Sources ===
[1] TEXT - DIN EN ISO 16090-1:2018-12
[2] TABLE - DIN EN ISO 16090-1:2018-12
[3] IMAGE - DIN EN ISO 16090-1:2018-12
```

## Testing

The project includes a comprehensive test suite. To run the tests:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_rag_evaluation.py
```

Note: The test suite is configured to work both in CI environments (using mocks) and locally (using real test data).

## Implementation Details

### Document Processing

The system uses specialized document processing techniques to handle multi-modal content:

1. **Text Extraction**: Standard text extraction from PDF documents
2. **Table Extraction**: Identification and extraction of tables using heuristics and OCR
3. **Image Extraction**: Extraction of images and identification of technical drawings
4. **Content-Aware Chunking**: Chunking that preserves the integrity of tables and images

### Vector Store

The multi-modal vector store uses specialized embedding models for different content types:

1. **Text Embeddings**: General-purpose text embedding model (e.g., all-MiniLM-L6-v2)
2. **Table Embeddings**: Specialized model for tabular data
3. **Image Embeddings**: Vision-language model for image content

### RAG Pipeline

The RAG pipeline integrates the multi-modal vector store with a language model:

1. **Query Analysis**: Determine the relevant content types for the query
2. **Multi-Modal Retrieval**: Retrieve relevant documents of each content type
3. **Response Generation**: Generate comprehensive answers based on retrieved documents

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
