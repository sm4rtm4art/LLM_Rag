# RAG Pipeline Factory and Builder Patterns

This document describes the Factory and Builder design patterns implemented for the RAG Pipeline module, which simplify component creation and pipeline construction.

## Core Design Patterns

The RAG pipeline module implements two key design patterns:

1. **Factory Pattern** (`component_factory.py`) - For creating and registering pipeline components
2. **Builder Pattern** (`pipeline_builder.py`) - For constructing complex pipelines with a fluent interface

## Factory Pattern Usage

The `RAGComponentFactory` allows you to create and register retriever, formatter, and generator components:

```python
from llm_rag.rag.pipeline import rag_factory

# Create a vector retriever
retriever = rag_factory.create_retriever(
    retriever_type="vector",
    source=my_vectorstore,
    top_k=5
)

# Create a markdown formatter
formatter = rag_factory.create_formatter(
    formatter_type="markdown",
    include_metadata=True,
    max_length=4000
)

# Create an LLM generator
generator = rag_factory.create_generator(
    generator_type="llm",
    llm=my_language_model,
    prompt_template="Answer based on this context: {context}\n\nQuestion: {query}",
    apply_anti_hallucination=True
)
```

### Registering Custom Components

You can register custom components to use with the factory:

```python
# Define a custom retriever
class MySpecialRetriever(BaseRetriever):
    def __init__(self, special_param=None):
        self.special_param = special_param

    def retrieve(self, query, **kwargs):
        # Custom retrieval logic
        return my_documents

# Register it with the factory
rag_factory.register_retriever("special", MySpecialRetriever)

# Use it later
retriever = rag_factory.create_retriever(
    retriever_type="special",
    special_param="custom_value"
)
```

## Builder Pattern Usage

The `RAGPipelineBuilder` provides a fluent interface for constructing pipelines:

```python
from llm_rag.rag.pipeline import create_rag_pipeline

# Create a standard RAG pipeline
pipeline = (
    create_rag_pipeline()
    .with_vector_retriever(my_vectorstore, top_k=3)
    .with_markdown_formatter(include_metadata=True)
    .with_llm_generator(
        llm=my_language_model,
        apply_anti_hallucination=True
    )
    .build()
)

# Use the pipeline
result = pipeline.process_query("What is RAG?")
```

### Conversational Pipelines

You can create conversational pipelines that maintain chat history:

```python
# Create a conversational RAG pipeline
conv_pipeline = (
    create_rag_pipeline()
    .with_conversational_pipeline()
    .with_vector_retriever(my_vectorstore)
    .with_markdown_formatter()
    .with_templated_generator(
        llm=my_language_model,
        templates={
            "default": "Context: {context}\n\nChat history: {history}\n\nUser: {query}\nAssistant:",
            "follow_up": "Based on the previous answer and this context: {context}\n\nUser: {query}\nAssistant:"
        }
    )
    .build()
)

# Use the conversational pipeline
response1 = conv_pipeline.process_query("What is RAG?")
response2 = conv_pipeline.process_query("What are its advantages?")  # Uses history
```

### Additional Configuration

Add custom configuration to your pipelines:

```python
pipeline = (
    create_rag_pipeline()
    .with_vector_retriever(my_vectorstore)
    .with_simple_formatter()
    .with_llm_generator(my_language_model)
    .with_config(
        verbose=True,
        cache_results=True,
        max_tokens=1000
    )
    .build()
)
```

## Testing

Both the factory and builder components are thoroughly tested. Run tests with:

```bash
pytest tests/rag/pipeline/test_factory_builder.py
```

## Design Benefits

These patterns provide several key benefits:

1. **Improved Maintainability**: Clean separation of concerns
2. **Enhanced Extensibility**: Easy to add new component types
3. **Better Testability**: Components can be tested in isolation
4. **Simplified Usage**: Intuitive API for pipeline construction
5. **Reduced Boilerplate**: Common component creation and configuration is abstracted away

## Available Components

### Retrievers

- `VectorStoreRetriever`: Uses vector database for semantic search
- `HybridRetriever`: Combines multiple retrieval strategies

### Formatters

- `SimpleContextFormatter`: Basic text formatting
- `MarkdownContextFormatter`: Formats context as markdown

### Generators

- `LLMGenerator`: Basic LLM response generation
- `TemplatedGenerator`: Supports multiple prompt templates
