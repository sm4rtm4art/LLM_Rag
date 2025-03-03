# LLM RAG Demo

This demo shows how to use a local LLM with a RAG (Retrieval-Augmented Generation) pipeline.

## Prerequisites

- Python 3.12+
- A local GGUF format LLM model (e.g., Llama 2, Mistral, etc.)
- Document data to ingest into the vector store

## Setup

1. **Install the dependencies**:

   ```bash
   pip install -e .
   ```

2. **Download a GGUF model**:
   Download a GGUF format model from [TheBloke on HuggingFace](https://huggingface.co/TheBloke) or another source, and place it in the `models` directory.

   Recommended models:

   - Llama 2 7B Chat (Q4_K_M): [Link](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf)
   - Mistral 7B Instruct (Q4_K_M): [Link](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)

3. **Ingest documents into the vector store**:
   ```bash
   python -m src.llm_rag.main --data-dir data/documents
   ```

## Usage

### Interactive Mode

```bash
python demo_llm_rag.py --model-path models/llama-2-7b-chat.Q4_K_M.gguf
```

### Query Mode

```bash
python demo_llm_rag.py --model-path models/llama-2-7b-chat.Q4_K_M.gguf --query "What is RAG?"
```

### GPU Acceleration

To use GPU acceleration (if available), specify the number of layers to offload to GPU:

```bash
python demo_llm_rag.py --model-path models/llama-2-7b-chat.Q4_K_M.gguf --n-gpu-layers 32
```

## Additional Options

- `--db-dir`: Path to the vector database directory (default: `data/vector_db`)
- `--collection-name`: Name of the collection in the vector database (default: `documents`)
- `--n-ctx`: Context size for the model (default: 2048)

## Troubleshooting

1. **Model not found**: Ensure you've downloaded a GGUF model and placed it in the correct location.
2. **Empty vector store**: Run the ingestion command to populate the vector store with documents.
3. **Out of memory**: Try a smaller model, reduce the context size, or reduce the number of GPU layers.

## Examples

### Question about RAG

```bash
python demo_llm_rag.py --query "What are the benefits of RAG systems?"
```

### Question about specific documents

```bash
python demo_llm_rag.py --query "Summarize the main points about DIN VDE standards"
```
