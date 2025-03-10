"""Base classes for the RAG pipeline.

This module contains the base classes used by the RAG pipeline components.
These are separated from other modules to avoid circular imports.
"""

from typing import Optional, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

# Define default templates
DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

{history}

Question: {query}

Answer:
"""

DEFAULT_CONVERSATIONAL_TEMPLATE = """
You are a helpful AI assistant. Answer the user's question based on the
provided context. If you don't know the answer, say so.

Context:
{context}

Conversation history:
{history}

Question: {query}

Answer:
"""


class BaseRAGPipeline:
    """Base class for RAG Pipelines."""

    def __init__(
        self,
        vectorstore: VectorStore,
        llm: BaseLanguageModel,
        top_k: int = 5,
        prompt_template: Union[str, PromptTemplate] = DEFAULT_PROMPT_TEMPLATE,
        history_size: int = 10,
    ):
        """Initialize the RAG pipeline.

        Args:
            vectorstore: Vector store for document retrieval
            llm: Language model for generating responses
            top_k: Number of documents to retrieve
            prompt_template: Template for formatting prompts to the LLM
            history_size: Maximum number of messages to keep in history

        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        self.history_size = history_size

        # Convert string template to PromptTemplate if needed
        if isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "history", "query"],
            )
        else:
            self.prompt_template = prompt_template


class BaseConversationalRAGPipeline(BaseRAGPipeline):
    """Base class for Conversational RAG Pipelines."""

    def __init__(
        self,
        vectorstore: VectorStore,
        llm: BaseLanguageModel,
        top_k: int = 3,
        history_size: int = 3,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
    ):
        """Initialize the conversational RAG pipeline.

        Args:
            vectorstore: Vector store for document retrieval
            llm: Language model for generating responses
            top_k: Number of documents to retrieve
            history_size: Maximum number of conversation turns to keep
            prompt_template: Optional custom prompt template

        """
        # Use conversational template if none provided
        if prompt_template is None:
            prompt_template = DEFAULT_CONVERSATIONAL_TEMPLATE

        super().__init__(
            vectorstore=vectorstore,
            llm=llm,
            top_k=top_k,
            prompt_template=prompt_template,
            history_size=history_size,
        )
        self.conversation_history = {}
