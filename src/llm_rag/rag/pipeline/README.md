# RAG Pipeline Module

This directory contains the modular components of the RAG (Retrieval-Augmented Generation) pipeline. The code has been refactored from a monolithic implementation into smaller, focused modules following SOLID principles.

## Module Structure

- **base.py**: Core RAGPipeline class and supporting utilities
- **conversational.py**: Conversational extension of the RAG pipeline
- **document_processor.py**: Document processing utilities
- **retrieval.py**: Document retrieval components and strategies
- **context.py**: Context formatting components and strategies
- **generation.py**: Response generation components and strategies

## Design Principles

The refactoring follows these key design principles:

1. **Single Responsibility Principle**: Each module and class has a clear, focused responsibility
2. **Open-Closed Principle**: Components can be extended without modifying existing code
3. **Liskov Substitution Principle**: Components can be swapped with their subtypes
4. **Interface Segregation Principle**: Clients depend only on interfaces they use
5. **Dependency Inversion**: High-level modules depend on abstractions, not details

## Component Architecture

### Retrieval Components

Retrieval components are responsible for fetching relevant documents from various sources:

- `BaseRetriever`: Abstract base class for all retrievers
- `DocumentRetriever`: Protocol defining the retriever interface
- `VectorStoreRetriever`: Retriever implementation for vector stores
- `HybridRetriever`: Combines multiple retrieval strategies

### Context Formatting

Context formatters prepare retrieved documents for use by the language model:

- `BaseContextFormatter`: Abstract base class for all formatters
- `ContextFormatter`: Protocol defining the formatter interface
- `SimpleContextFormatter`: Basic text-based document formatter
- `MarkdownContextFormatter`: Markdown-based document formatter

### Generation Components

Generation components handle producing responses based on the context:

- `BaseGenerator`: Abstract base class for all generators
- `ResponseGenerator`: Protocol defining the generator interface
- `LLMGenerator`: LLM-based response generator
- `TemplatedGenerator`: Supports multiple prompt templates

## Factory Functions

To simplify component creation, the module provides factory functions:

- `create_retriever()`: Creates appropriate retriever based on the source
- `create_formatter()`: Creates the specified formatter type
- `create_generator()`: Creates the specified generator type

## Backward Compatibility

The original `pipeline.py` file is maintained for backward compatibility, re-exporting all components from the modular implementation. This approach allows gradual migration to the new structure while ensuring existing code continues to work.

## Usage Examples

### Using the modular components directly:

```python
from llm_rag.rag.pipeline.retrieval import create_retriever
from llm_rag.rag.pipeline.context import create_formatter
from llm_rag.rag.pipeline.generation import create_generator

# Create components
retriever = create_retriever(source=vectorstore, top_k=5)
formatter = create_formatter(format_type="markdown", include_metadata=True)
generator = create_generator(llm=llm, apply_anti_hallucination=True)

# Use the pipeline
documents = retriever.retrieve("What is RAG?")
context = formatter.format_context(documents)
response = generator.generate(query="What is RAG?", context=context)
```

### Using the integrated pipeline:

```python
from llm_rag.rag.pipeline import RAGPipeline

# Create pipeline
pipeline = RAGPipeline(
    vectorstore=vectorstore,
    llm=llm,
    top_k=5,
)

# Process a query
result = pipeline.query("What is RAG?")
print(result["response"])
```
