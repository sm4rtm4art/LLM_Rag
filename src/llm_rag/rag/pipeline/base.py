#!/usr/bin/env python
"""Base RAG pipeline implementation.

This module provides the core implementation of the RAG (Retrieval-Augmented Generation)
pipeline, including the base RAGPipeline class and supporting classes.
"""

from typing import Any, Dict, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

from llm_rag.rag.pipeline.context import create_formatter
from llm_rag.rag.pipeline.generation import create_generator
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


class RAGPipeline:
    """Base RAG pipeline implementation.

    This class implements the core RAG (Retrieval-Augmented Generation) pipeline,
    which combines document retrieval with language model generation to provide
    context-aware responses.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        llm: BaseLanguageModel,
        top_k: int = 4,
        prompt_template: Union[str, PromptTemplate] = DEFAULT_PROMPT_TEMPLATE,
        history_size: int = 10,
    ):
        """Initialize the RAG pipeline.

        Args:
            vectorstore: Vector store for document retrieval
            llm: Language model for response generation
            top_k: Number of documents to retrieve for each query
            prompt_template: Template for generating prompts
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

        # Initialize pipeline components
        self._retriever = create_retriever(vectorstore, top_k=top_k)
        self._formatter = create_formatter()
        self._generator = create_generator(llm, self.prompt_template)

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
