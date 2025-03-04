# Demo Scripts

This directory contains various demo scripts that showcase different aspects of the LLM RAG system.

## Available Demos

### 1. Multi-Modal RAG Demo

The `demo_din_multimodal_rag.py` script demonstrates the multi-modal RAG system with DIN standards:

```bash
python demos/demo_din_multimodal_rag.py --din_path /path/to/din/standards
```

### 2. Hugging Face Model Demo

The `demo_huggingface.py` script demonstrates using Hugging Face models with the RAG system:

```bash
python demos/demo_huggingface.py
```

### 3. Llama Model Demo

The `demo_llama_rag.py` script demonstrates using Llama models with the RAG system:

```bash
python demos/demo_llama_rag.py
```

### 4. Simple Document Loading Demo

The `simple_load_documents.py` script demonstrates how to load and process documents:

```bash
python demos/simple_load_documents.py --input_dir /path/to/documents
```

## Command-line Options

Most demo scripts support the following command-line options:

- `--input_dir`: Path to input documents directory
- `--model`: Name or path of the model to use (default: "microsoft/phi-2")
- `--device`: Device to use for model inference ("cpu", "cuda", "auto") (default: "auto")
- `--persist_dir`: Directory to persist the vector store (default: "chroma_db")
- `--top_k`: Number of documents to retrieve (default: 3)

For specific options for each demo, run:

```bash
python demos/<demo_script>.py --help
```

## Example Usage

Here's an example of running the Hugging Face demo with custom parameters:

```bash
python demos/demo_huggingface.py --model "google/flan-t5-base" --device "cpu" --top_k 5
```
