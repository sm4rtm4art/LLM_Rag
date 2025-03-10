# RAG Anti-Hallucination Utilities

This module provides utilities to detect and mitigate hallucinations in Retrieval-Augmented Generation (RAG) systems. It combines two complementary approaches:

1. **Entity-based verification**: Checks if entities in the generated response are present in the source context
2. **Embedding-based verification**: Computes semantic similarity between the response and context

## Features

- Configuration-based approach for easy customization
- Multiple language support for stopwords
- Detailed metadata about verification results
- Human review flagging for critical cases
- Backward compatibility with simpler entity-based approaches

## Installation

The basic module has minimal dependencies, but for embedding-based verification, additional packages are required:

```bash
# Core functionality
pip install typing-extensions

# For embedding-based verification (optional)
pip install sentence-transformers scikit-learn numpy
```

## Basic Usage

```python
from llm_rag.rag.anti_hallucination import post_process_response

# Get a response from your LLM/RAG system
response = "Einstein developed the theory of general relativity in 1915."
context = "Albert Einstein published his general theory of relativity in 1915."

# Post-process the response to add warnings about potential hallucinations
processed_response = post_process_response(response, context)

# Output will either be the original response or will include warning messages
# about potential hallucinations
print(processed_response)
```

## Advanced Usage

### Using the Configuration Object

```python
from llm_rag.rag.anti_hallucination import post_process_response, HallucinationConfig

# Create a configuration with custom thresholds
config = HallucinationConfig(
    entity_threshold=0.8,  # Stricter entity verification
    embedding_threshold=0.7,
    model_name="all-MiniLM-L6-v2",  # Use a different embedding model
    flag_for_human_review=True,  # Enable human review flagging
    human_review_threshold=0.6  # Threshold for human review
)

# Process response with the configuration
processed_response = post_process_response(
    response,
    context,
    config=config
)
```

### Getting Metadata

```python
from llm_rag.rag.anti_hallucination import post_process_response

# Process response and get both the processed text and metadata
processed_response, metadata = post_process_response(
    response,
    context,
    return_metadata=True
)

# The metadata contains detailed information about the verification process
print(f"Verified: {metadata['verified']}")
print(f"Entity coverage: {metadata['entity_coverage']:.2f}")
print(f"Embedding similarity: {metadata['embedding_similarity']:.2f}")
print(f"Hallucination score: {metadata['hallucination_score']:.2f}")
print(f"Human review recommended: {metadata['human_review_recommended']}")
print(f"Missing entities: {metadata['missing_entities']}")
```

### Multi-Language Support

```python
from llm_rag.rag.anti_hallucination import post_process_response

# Process response with multiple languages for stopwords
processed_response = post_process_response(
    response,
    context,
    languages=["en", "de"]  # English and German stopwords
)
```

### Human Review Integration

```python
from llm_rag.rag.anti_hallucination import post_process_response

# Enable human review flagging
processed_response, metadata = post_process_response(
    response,
    context,
    flag_for_human_review=True,
    return_metadata=True
)

# Send to human review queue if needed
if metadata["human_review_recommended"]:
    human_review_queue.add_item({
        "original_response": response,
        "processed_response": processed_response,
        "context": context,
        "hallucination_score": metadata["hallucination_score"],
        "missing_entities": metadata["missing_entities"]
    })

    # Inform the user that an expert will review this response
    print("Your response is being reviewed by our experts.")
```

## Customizing Stopwords

The module loads stopwords from configuration files if available, or falls back to defaults:

```python
# Custom stopwords file structure (JSON format)
# Store in llm_rag/rag/resources/stopwords_en.json

["a", "an", "the", "and", "or", "but", "if", "then", "else", ...]
```

## Performance Considerations

- The SentenceTransformer models are cached to avoid reloading
- Entity extraction is optimized for speed
- Embedding-based verification is skipped if dependencies are not available

## Integration with RAG Pipeline

Here's how to integrate the anti-hallucination checks into a typical RAG pipeline:

```python
def rag_pipeline(query, documents):
    # 1. Retrieve relevant context from documents
    context = retrieve_context(query, documents)

    # 2. Generate a response using an LLM
    response = generate_response(query, context)

    # 3. Verify and post-process the response
    config = HallucinationConfig(
        flag_for_human_review=True,
        use_embeddings=True
    )
    processed_response, metadata = post_process_response(
        response, context, config=config, return_metadata=True
    )

    # 4. Log verification results
    log_verification_results(query, response, metadata)

    return processed_response
```

## Testing

Run the test suite to ensure everything is working properly:

```bash
python -m unittest llm_rag.tests.test_anti_hallucination
```
