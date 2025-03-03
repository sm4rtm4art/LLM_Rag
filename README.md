# LLM RAG System

[![codecov](https://codecov.io/gh/sm4rtm4art/llm-rag/branch/main/graph/badge.svg)](https://codecov.io/gh/sm4rtm4art/llm-rag)

A Retrieval-Augmented Generation (RAG) system that uses LLMs to answer questions based on your documents.

## Features

- Document loading and processing
- Vector database storage
- Conversational RAG pipeline
- Command-line interface
- Support for various LLM backends (HuggingFace, LLaMA)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-rag.git
cd llm-rag

# Install the package
pip install -e .
```

## Quick Start with Local LLM Demo

1. **Download a model** using our convenient script:

   ```bash
   python download_model.py
   ```

2. **Ingest your documents**:

   ```bash
   python -m src.llm_rag.main --data-dir data/documents
   ```

3. **Run the demo in interactive mode**:
   ```bash
   python demo_llm_rag.py
   ```

See [README_DEMO.md](README_DEMO.md) for more detailed instructions and options.

## Usage

### Loading Documents

```bash
python -m scripts.load_documents --input-dir data/documents --output-dir data/vector_db --collection-name my_docs
```

### RAG CLI

```bash
# Basic usage
python -m scripts.rag_cli --vector-db data/vector_db --collection-name my_docs

# Use a different model
python -m scripts.rag_cli --vector-db data/vector_db --collection-name my_docs --model google/flan-t5-large

# Use without device_map (if you have issues with accelerate)
python -m scripts.rag_cli --vector-db data/vector_db --collection-name my_docs --no-device-map

# Use LLaMA model
python -m scripts.rag_cli --vector-db data/vector_db --collection-name my_docs --use-llama --llama-model-path path/to/model.gguf
```

## Troubleshooting

### Model Loading Issues

If you encounter issues with model loading:

1. Try using the `--no-device-map` option, which doesn't require the `accelerate` package:

   ```bash
   python -m scripts.rag_cli --vector-db data/vector_db --collection-name my_docs --no-device-map
   ```

2. For LLaMA models, ensure you have the `llama-cpp-python` package installed:
   ```bash
   pip install llama-cpp-python
   ```

## License

MIT
