"""RAG pipeline components for the LLM-RAG system.

This module provides components for building retrieval-augmented generation
pipelines. It includes classes for document retrieval, context formatting,
query generation, and response post-processing.

The original monolithic implementation has been broken down into focused
modules while maintaining backward compatibility through this package.
"""

# Import and re-export all components for easy access
# These will be available via llm_rag.rag.pipeline

# Base class definitions
# Base components
from llm_rag.rag.pipeline.base import (
    InMemoryChatMessageHistory,
    MyCustomDocument,
    RAGPipeline,
)
from llm_rag.rag.pipeline.base_classes import (
    DEFAULT_CONVERSATIONAL_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    BaseConversationalRAGPipeline,
    BaseRAGPipeline,
)

# Context formatting components
from llm_rag.rag.pipeline.context import (
    BaseContextFormatter,
    ContextFormatter,
    MarkdownContextFormatter,
    SimpleContextFormatter,
    create_formatter,
)

# Conversational pipeline components
from llm_rag.rag.pipeline.conversational import (
    ConversationalRAGPipeline,
)

# Document processing components
from llm_rag.rag.pipeline.document_processor import (
    _process_document,
    _process_documents,
)
from llm_rag.rag.pipeline.generation import (
    DEFAULT_PROMPT_TEMPLATE as GENERATION_PROMPT_TEMPLATE,
)

# Generation components
from llm_rag.rag.pipeline.generation import (
    BaseGenerator,
    LLMGenerator,
    ResponseGenerator,
    TemplatedGenerator,
    create_generator,
)

# Retrieval components
from llm_rag.rag.pipeline.retrieval import (
    BaseRetriever,
    DocumentRetriever,
    HybridRetriever,
    VectorStoreRetriever,
    create_retriever,
)

# Define what's available when using `from llm_rag.rag.pipeline import *`
__all__ = [
    # Base components
    "DEFAULT_PROMPT_TEMPLATE",
    "DEFAULT_CONVERSATIONAL_TEMPLATE",
    "BaseRAGPipeline",
    "BaseConversationalRAGPipeline",
    "InMemoryChatMessageHistory",
    "MyCustomDocument",
    "RAGPipeline",
    # Conversational components
    "ConversationalRAGPipeline",
    # Document processing functions
    "_process_document",
    "_process_documents",
    # Retrieval components
    "BaseRetriever",
    "DocumentRetriever",
    "HybridRetriever",
    "VectorStoreRetriever",
    "create_retriever",
    # Context formatting components
    "BaseContextFormatter",
    "ContextFormatter",
    "MarkdownContextFormatter",
    "SimpleContextFormatter",
    "create_formatter",
    # Generation components
    "BaseGenerator",
    "GENERATION_PROMPT_TEMPLATE",
    "LLMGenerator",
    "ResponseGenerator",
    "TemplatedGenerator",
    "create_generator",
]
