"""Pipeline module for the RAG system.

This module provides the core pipeline components for the RAG system,
including document retrieval, context formatting, and response generation.
"""

# Import and re-export all components for easy access
# These will be available via llm_rag.rag.pipeline
# Base class definitions
# Base components
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

from .base import RAGPipeline

# Context formatting components
from .base_classes import (
    DEFAULT_CONVERSATIONAL_TEMPLATE,
    DEFAULT_PROMPT_TEMPLATE,
    BaseConversationalRAGPipeline,
    BaseRAGPipeline,
)
from .context import BaseContextFormatter as BaseFormatter
from .context import MarkdownContextFormatter as MarkdownFormatter
from .context import SimpleContextFormatter as SimpleFormatter
from .conversational import ConversationalRAGPipeline
from .generation import BaseGenerator
from .generation import LLMGenerator as SimpleGenerator
from .generation import TemplatedGenerator as StreamingGenerator
from .retrieval import BaseRetriever, HybridRetriever
from .retrieval import VectorStoreRetriever as SimpleRetriever

# Define HTMLFormatter as an alias for MarkdownFormatter for now
HTMLFormatter = MarkdownFormatter


class MessageHistory:
    """Base class for message history."""

    def __init__(self):
        """Initialize an empty message history."""
        self.messages = []

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        self.messages.append(message)

    def get_messages(self) -> list[BaseMessage]:
        """Get all messages in the history."""
        return self.messages

    def clear(self) -> None:
        """Clear the message history."""
        self.messages = []


# Re-export all components
__all__ = [
    # Base components
    "BaseRAGPipeline",
    "BaseConversationalRAGPipeline",
    "BaseMessage",
    "BaseMessageHistory",
    "BaseLanguageModel",
    "PromptTemplate",
    "VectorStore",
    # Pipeline implementations
    "RAGPipeline",
    "ConversationalRAGPipeline",
    # Retrieval components
    "BaseRetriever",
    "SimpleRetriever",
    "HybridRetriever",
    # Generation components
    "BaseGenerator",
    "SimpleGenerator",
    "StreamingGenerator",
    # Context formatting components
    "BaseFormatter",
    "SimpleFormatter",
    "MarkdownFormatter",
    "HTMLFormatter",
    # Templates
    "DEFAULT_PROMPT_TEMPLATE",
    "DEFAULT_CONVERSATIONAL_TEMPLATE",
]
