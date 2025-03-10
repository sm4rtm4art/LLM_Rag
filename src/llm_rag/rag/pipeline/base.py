#!/usr/bin/env python
"""Base RAG pipeline implementation.

This module provides the core implementation of the RAG (Retrieval-Augmented Generation)
pipeline, including the base RAGPipeline class and supporting classes.
"""

from collections.abc import MutableMapping
from typing import Any, Dict, Iterator, Union

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

from llm_rag.rag.pipeline.context import create_formatter
from llm_rag.rag.pipeline.generation import create_generator

# Import helper functions to avoid circular imports inside the init method
from llm_rag.rag.pipeline.retrieval import create_retriever
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)

# Default prompt template for RAG
DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

{history}

Question: {query}

Answer:"""


# Define a simple in-memory chat message history class
class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """Simple in-memory implementation of chat message history.

    This class stores chat messages in memory and is useful for maintaining
    conversation context during a RAG session.
    """

    def __init__(self):
        """Initialize an empty message history."""
        self.messages = []

    def add_message(self, message):
        """Add a message to the history.

        Args:
            message: The message to add to the history

        """
        self.messages.append(message)

    def clear(self):
        """Clear all messages from the history."""
        self.messages = []


class MyCustomDocument(MutableMapping[str, Any]):
    """Custom document class to standardize document handling.

    This class provides a consistent interface for documents from various
    sources with different data structures.
    """

    def __init__(self, content: str, metadata: Dict[Any, Any]):
        """Initialize a custom document.

        Args:
            content: The document content as text
            metadata: Associated metadata for the document

        """
        self._content = content
        self._metadata = metadata
        self._store = {"page_content": content, "metadata": metadata}

    def __getitem__(self, key: str) -> Any:
        """Get an item by key.

        Args:
            key: The key to look up

        Returns:
            The value associated with the key

        Raises:
            KeyError: If the key is not found

        """
        return self._store[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an item by key.

        Args:
            key: The key to set
            value: The value to set

        """
        self._store[key] = value

        # Update underlying attributes if setting specific keys
        if key == "page_content":
            self._content = value
        elif key == "metadata":
            self._metadata = value

    def __delitem__(self, key: str) -> None:
        """Delete an item by key.

        Args:
            key: The key to delete

        Raises:
            KeyError: If the key is not found

        """
        del self._store[key]

        # Reset underlying attributes if deleting specific keys
        if key == "page_content":
            self._content = ""
        elif key == "metadata":
            self._metadata = {}

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the keys.

        Returns:
            An iterator over the keys in the document

        """
        return iter(self._store)

    def __len__(self) -> int:
        """Return the number of items.

        Returns:
            The number of items in the document

        """
        return len(self._store)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary.

        Returns:
            A dictionary representation of the document

        """
        return dict(self)


class RAGPipeline:
    """Base RAG (Retrieval-Augmented Generation) pipeline.

    This class implements the core RAG functionality, including document retrieval,
    context formatting, and response generation. It follows a modular design with
    separate methods for each step in the pipeline.
    """

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
        self.conversation_history = {}

        # Convert string template to PromptTemplate if needed
        if isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "history", "query"],
            )
        else:
            self.prompt_template = prompt_template

        # Create modular components
        self._retriever = create_retriever(source=vectorstore, top_k=top_k)
        self._formatter = create_formatter(format_type="simple", include_metadata=True)
        self._generator = create_generator(llm=llm, prompt_template=prompt_template, apply_anti_hallucination=True)

        logger.info(
            "Initialized RAGPipeline with top_k=%d, history_size=%d",
            top_k,
            history_size,
        )

    def query(self, query: str) -> Dict[str, Any]:
        """Process a query through the RAG pipeline.

        This method orchestrates the entire RAG process: retrieval, context formatting,
        and response generation.

        Args:
            query: The query to process

        Returns:
            Dictionary containing the query, response, and retrieved documents

        """
        # Retrieve documents
        documents = self._retriever.retrieve(query)

        # Format context
        context = self._formatter.format_context(documents)

        # Generate response
        response = self._generator.generate(query=query, context=context)

        # Return results
        return {
            "query": query,
            "response": response,
            "documents": documents,
            "context": context,
        }
