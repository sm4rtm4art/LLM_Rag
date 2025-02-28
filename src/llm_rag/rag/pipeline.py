"""RAG pipeline implementation for the llm-rag package.

This module provides the main pipeline for retrieval-augmented generation.
"""
from typing import Any, Dict, List, Optional, Union

from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    This class implements the core RAG functionality, combining document
    retrieval with language model generation.
    """

    def __init__(
        self,
        vectorstore: Any,  # Will be more specific once we have a VectorStore interface
        llm: BaseLLM,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        top_k: int = 3,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
        ----
            vectorstore: Vector store for document retrieval.
            llm: Language model for generation.
            prompt_template: Template for constructing prompts with retrieved
                context. If a string, it should contain {context} and {query}
                placeholders.
            top_k: Number of documents to retrieve.

        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k

        # Set up prompt template
        if prompt_template is None:
            self.prompt_template = PromptTemplate(
                input_variables=["context", "query"],
                template=(
                    "Answer the following query based on the provided "
                    "context:\n\n"
                    "Context:\n{context}\n\n"
                    "Query: {query}\n\n"
                    "Answer:"
                ),
            )
        elif isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(
                input_variables=["context", "query"],
                template=prompt_template,
            )
        else:
            self.prompt_template = prompt_template

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from the vector store.

        Args:
        ----
            query: Query string to search for.

        Returns:
        -------
            List of relevant document dictionaries.

        """
        results = self.vectorstore.search(query, n_results=self.top_k)
        # Ensure we return a List[Dict[str, Any]] type
        return [doc for doc in results]

    def generate(self, query: str, context: str) -> str:
        """Generate a response using the language model.

        Args:
        ----
            query: User query.
            context: Retrieved context to inform the response.

        Returns:
        -------
            Generated response.

        """
        prompt = self.prompt_template.format(context=context, query=query)
        response = self.llm.invoke(prompt)
        # Ensure we return a string
        return str(response)

    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a context string.

        Args:
        ----
            documents: List of document dictionaries.

        Returns:
        -------
            Formatted context string.

        """
        # Extract content from documents and join with separators
        doc_contents = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            if content:
                doc_contents.append(f"Document {i+1}:\n{content}")

        return "\n\n".join(doc_contents)

    def query(self, query: str) -> Dict[str, Any]:
        """Execute the complete RAG pipeline.

        Args:
        ----
            query: User query.

        Returns:
        -------
            Dictionary containing the query, retrieved documents,
            and generated response.

        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)

        # Format context from retrieved documents
        context = self.format_context(retrieved_docs)

        # Generate response
        response = self.generate(query, context)

        # Return combined result
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "response": response,
        }


class ConversationalRAGPipeline(RAGPipeline):
    """Conversational RAG pipeline with chat history.

    This class extends the basic RAG pipeline to support conversational
    interactions by maintaining chat history.
    """

    def __init__(
        self,
        vectorstore: Any,
        llm: BaseLLM,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        top_k: int = 3,
        history_size: int = 5,
    ) -> None:
        """Initialize the conversational RAG pipeline.

        Args:
        ----
            vectorstore: Vector store for document retrieval.
            llm: Language model for generation.
            prompt_template: Template for constructing prompts. Should contain
                {context}, {query}, and {history} placeholders.
            top_k: Number of documents to retrieve.
            history_size: Maximum number of conversation turns to keep in
                history.

        """
        # Set up default prompt template for conversational RAG
        if prompt_template is None:
            prompt_template = PromptTemplate(
                input_variables=["context", "query", "history"],
                template=(
                    "Answer the following query based on the provided context "
                    "and conversation history:\n\n"
                    "Context:\n{context}\n\n"
                    "Conversation history:\n{history}\n\n"
                    "Query: {query}\n\n"
                    "Answer:"
                ),
            )

        super().__init__(vectorstore, llm, prompt_template, top_k)
        self.history_size = history_size
        self.conversation_history: List[Dict[str, str]] = []

    def generate(self, query: str, context: str) -> str:
        """Generate a response using history and context.

        Args:
        ----
            query: User query.
            context: Retrieved context to inform the response.

        Returns:
        -------
            Generated response.

        """
        # Format history
        history_text = self.format_history()

        # Generate response including history
        prompt = self.prompt_template.format(context=context, query=query, history=history_text)
        response = self.llm.invoke(prompt)

        # Ensure we return a string
        response_str = str(response)

        # Update history
        self.add_to_history(query, response_str)

        return response_str

    def format_history(self) -> str:
        """Format conversation history into a string.

        Returns
        -------
            Formatted history string.

        """
        if not self.conversation_history:
            return "No conversation history."

        formatted_turns = []
        for turn in self.conversation_history:
            user_msg = turn.get("user", "")
            assistant_msg = turn.get("assistant", "")
            formatted_turns.append(f"User: {user_msg}\nAssistant: {assistant_msg}")

        return "\n\n".join(formatted_turns)

    def add_to_history(self, user_query: str, assistant_response: str) -> None:
        """Add a new turn to the conversation history.

        Args:
        ----
            user_query: User's query.
            assistant_response: Assistant's response.

        """
        self.conversation_history.append(
            {
                "user": user_query,
                "assistant": assistant_response,
            }
        )

        # Truncate history if needed
        if len(self.conversation_history) > self.history_size:
            self.conversation_history = self.conversation_history[-self.history_size :]

    def reset_history(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []

    def query(self, query: str) -> Dict[str, Any]:
        """Execute the complete conversational RAG pipeline.

        Args:
        ----
            query: User query.

        Returns:
        -------
            Dictionary containing the query, retrieved documents,
            and generated response.

        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)

        # Format context from retrieved documents
        context = self.format_context(retrieved_docs)

        # Generate response (this will also update history)
        response = self.generate(query, context)

        # Return combined result
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "response": response,
            "history": self.conversation_history,
        }
