"""RAG Pipeline implementation.

This module contains the implementation of the RAG pipeline, which combines
retrieval and generation to answer questions based on a knowledge base.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union

# TODO: This import generates a deprecation warning. When updating to newer
# LangChain versions, follow the migration guide at:
# https://python.langchain.com/docs/versions/migrating_memory/
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG Pipeline.

    This class implements a basic RAG pipeline that can answer
    questions based on a knowledge base.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        llm: BaseLanguageModel,
        top_k: int = 3,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
    ):
        """Initialize the RAG pipeline.

        Args:
        ----
            vectorstore: The vector store containing the document embeddings.
            llm: The language model to use for generation.
            top_k: The number of documents to retrieve.
            prompt_template: Optional custom prompt template.

        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k

        # Set up prompt template
        if prompt_template is None:
            self.prompt_template = PromptTemplate(
                input_variables=["context", "query"],
                template=(
                    "Answer the question based on the following context:\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {query}\n\n"
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

        logger.info("Initialized RAGPipeline")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from the vectorstore.

        Args:
        ----
            query: The query to search for.

        Returns:
        -------
            List of retrieved documents.

        """
        docs = None

        # Use search method from vectorstore
        try:
            # Call the search method directly with expected parameters
            docs = self.vectorstore.search(query, n_results=self.top_k)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

        # If docs is None, return empty list
        if docs is None:
            return []

        # Process the documents to ensure they're in the right format
        result = []

        # Check if docs is a list
        if not isinstance(docs, list):
            logger.warning(f"Expected list but got {type(docs)}")
            return []

        # Process each document
        for doc in docs:
            try:
                if isinstance(doc, dict):
                    # Already a dictionary
                    result.append(doc)
                elif hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                    # Document-like object
                    result.append(
                        {"content": getattr(doc, "page_content", ""), "metadata": getattr(doc, "metadata", {})}
                    )
                else:
                    # Try to convert to a dictionary
                    logger.warning(f"Unexpected document type: {type(doc)}")
                    result.append({"content": str(doc), "metadata": {}})
            except Exception as e:
                logger.warning(f"Error processing document: {e}")
                continue

        return result

    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents into a context string.

        Args:
        ----
            documents: List of documents to format.

        Returns:
        -------
            Formatted context string.

        """
        valid_docs = []
        for i, doc in enumerate(documents):
            if "content" in doc and doc["content"]:
                valid_docs.append(f"Document {i+1}:\n{doc['content']}")

        if not valid_docs:
            return "No relevant documents found."

        return "\n\n".join(valid_docs)

    def generate(self, query: str, context: str) -> str:
        """Generate a response based on the query and context.

        Args:
        ----
            query: The query to answer.
            context: The context to use for answering.

        Returns:
        -------
            Generated response.

        """
        prompt = self.prompt_template.format(
            context=context,
            query=query,
        )
        response = self.llm.invoke(prompt)
        # Ensure we return a string
        if isinstance(response, str):
            return response
        # Handle potential non-string responses from different LLM
        # implementations
        elif hasattr(response, "content"):
            return str(response.content)
        else:
            return str(response)

    def query(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query through the RAG pipeline.

        Args:
        ----
            query: The query to process.
            conversation_id: Optional ID for the conversation.

        Returns:
        -------
            Dictionary with query, response, and source documents.

        """
        # Retrieve relevant documents
        docs = self.retrieve(query)

        # Format documents into context string
        context = self.format_context(docs)

        # Generate response using the LLM
        response = self.generate(query, context)

        # Return response with source documents
        return {
            "query": query,
            "response": response,
            "source_documents": docs,
        }


class ConversationalRAGPipeline(RAGPipeline):
    """RAG pipeline with conversation history.

    This pipeline extends the basic RAG pipeline with conversation history
    tracking.
    """

    def __init__(
        self,
        vectorstore: Any,
        llm_chain: Any,
        top_k: int = 3,
        memory: Optional[Any] = None,
        history_size: int = 3,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
    ):
        """Initialize the conversational RAG pipeline.

        Args:
        ----
            vectorstore: Vector store for document retrieval.
            llm_chain: Language model chain for generating responses.
            top_k: Number of documents to retrieve.
            memory: Optional memory for storing conversation history.
            history_size: Maximum number of conversation turns to keep.
            prompt_template: Optional custom prompt template.

        """
        # Initialize with a placeholder LLM that will be replaced
        super().__init__(vectorstore, llm_chain, top_k)

        # Replace the LLM with the LLM chain
        self.llm = llm_chain

        # Set up conversational prompt template
        if prompt_template is None:
            self.prompt_template = PromptTemplate(
                input_variables=["context", "query", "history"],
                template=(
                    "Answer the question based on the following context "
                    "and conversation history:\n\n"
                    "Context:\n{context}\n\n"
                    "Conversation History:\n{history}\n\n"
                    "Question: {query}\n\n"
                    "Answer:"
                ),
            )
        elif isinstance(prompt_template, str):
            self.prompt_template = PromptTemplate(
                input_variables=["context", "query", "history"],
                template=prompt_template,
            )
        else:
            self.prompt_template = prompt_template

        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        self.conversations: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, str]] = []
        self.history_size = history_size

        logger.info("Initialized ConversationalRAGPipeline")

    def add_to_history(self, query: str, response: str, conversation_id: Optional[str] = None) -> None:
        """Add a query-response pair to the conversation history.

        Args:
        ----
            query: The user query.
            response: The system response.
            conversation_id: Optional ID for the conversation.

        """
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
            )

        self.conversations[conversation_id].chat_memory.add_user_message(query)
        self.conversations[conversation_id].chat_memory.add_ai_message(response)

        # Also update the conversation_history list for backward compatibility
        self.conversation_history.append({"user": query, "assistant": response})

        # Truncate history to the specified size
        if len(self.conversation_history) > self.history_size:
            self.conversation_history = self.conversation_history[-self.history_size :]

    def generate(self, query: str, context: str, history: str = "") -> str:
        """Generate a response based on the query, context, and history.

        Args:
        ----
            query: The query to answer.
            context: The context to use for answering.
            history: Optional conversation history.

        Returns:
        -------
            Generated response.

        """
        # If no history is provided, format it from the conversation history
        if not history:
            history = self.format_history()

        # Format the prompt with context, query, and history
        prompt = self.prompt_template.format(
            context=context,
            query=query,
            history=history,
        )

        # Use the LLM chain to generate a response
        response = self.llm.predict(prompt)

        # Ensure response is a string
        response_str = ""
        try:
            if isinstance(response, str):
                response_str = response
            elif hasattr(response, "content"):
                response_str = str(response.content)
            else:
                response_str = str(response)
        except Exception as e:
            logger.error(f"Error converting response to string: {e}")
            response_str = "Error generating response"

        # Add the query and response to the history
        self.add_to_history(query, response_str)

        return response_str

    def query(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query with conversation history.

        Args:
        ----
            query: The query to process.
            conversation_id: Optional ID for the conversation.

        Returns:
        -------
            Dictionary with query, response, and source documents.

        """
        # Generate a conversation ID if not already tracking this conversation
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        # Retrieve relevant documents
        docs = self.retrieve(query)

        # Format documents into context string
        context = self.format_context(docs)

        # Generate response using the LLM (this also adds to history)
        response = self.generate(query, context)

        # Return response with source documents
        return {
            "query": query,
            "response": response,
            "source_documents": docs,
            "retrieved_documents": docs,  # For backward compatibility
            "conversation_id": conversation_id,
            "history": self.conversation_history,
        }

    def reset_history(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
        self.memory.clear()
        logger.info("Conversation memory reset")

    def format_history(self) -> str:
        """Format conversation history into a string.

        Returns
        -------
            Formatted history string.

        """
        if not self.conversation_history:
            return "No conversation history."

        history_parts = []
        for turn in self.conversation_history:
            history_parts.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")

        return "\n\n".join(history_parts)

    def _process_retrieved_documents(
        self, source_documents: List[Union[Document, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Process retrieved documents to a standard format.

        Args:
        ----
            source_documents: List of retrieved documents.

        Returns:
        -------
            List of processed documents.

        """
        retrieved_documents: List[Dict[str, Any]] = []

        # Handle empty input
        if not source_documents:
            return retrieved_documents

        # Process each document
        for doc in source_documents:
            try:
                if isinstance(doc, dict):
                    # Handle dictionary format
                    content = doc.get("content", "")
                    if isinstance(content, str) and len(content) > 200:
                        content = content[:200] + "..."
                    retrieved_documents.append({"content": content, "metadata": doc.get("metadata", {})})
                elif hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                    # Handle Document objects
                    content = getattr(doc, "page_content", "")
                    if isinstance(content, str) and len(content) > 200:
                        content = content[:200] + "..."
                    retrieved_documents.append({"content": content, "metadata": getattr(doc, "metadata", {})})
                else:
                    # Skip invalid documents
                    logger.warning(f"Skipping document with unexpected type: {type(doc)}")
            except Exception as e:
                logger.error(f"Error processing document: {e}")

        return retrieved_documents
