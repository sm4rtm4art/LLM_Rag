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

## CI/CD Integration

These demo scripts are automatically tested in our CI/CD pipeline to ensure they continue to work as the codebase evolves.

### Automated Testing

The CI/CD pipeline runs basic functionality tests on these demos to verify:

1. They can be imported without errors
2. They can be executed with default parameters
3. They produce expected output formats

### Docker Integration

You can also run these demos inside the Docker container:

```bash
# Build the Docker image
docker build -t llm-rag .

# Run a demo script
docker run -v $(pwd)/data:/app/data llm-rag python -m demos.demo_huggingface --input_dir /app/data
```

## Kubernetes Integration

For running demos in a Kubernetes environment, we provide a job template in the `k8s` directory:

```bash
# Customize the job template
cp k8s/demo-job-template.yaml k8s/my-demo-job.yaml
# Edit my-demo-job.yaml to specify which demo to run

# Apply the job
kubectl apply -f k8s/my-demo-job.yaml
```

## Troubleshooting

If you encounter issues with the demos:

1. Verify you have the required dependencies installed:

   ```bash
   uv pip install -e ".[dev]"
   ```

2. Check that you have the necessary models downloaded (if using local models)

3. Ensure your input data is in the correct format

4. For GPU acceleration, verify that your CUDA environment is properly set up
