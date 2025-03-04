"""RAG Pipeline implementation.

This module contains the implementation of the RAG pipeline, which combines
retrieval and generation to answer questions based on a knowledge base.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
    ConversationalRetrievalChain,
)

# TODO: This import generates a deprecation warning. When updating to newer
# LangChain versions, follow the migration guide at:
# https://python.langchain.com/docs/versions/migrating_memory/
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class ConversationalRAGPipeline:
    """Conversational RAG Pipeline.

    This class implements a conversational RAG pipeline that can answer
    questions based on a knowledge base, while maintaining conversation
    context.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        llm: BaseLanguageModel,
        top_k: int = 3,
        memory_key: str = "chat_history",
    ):
        """Initialize the RAG pipeline.

        Args:
        ----
            vectorstore: The vector store containing the document embeddings.
            llm: The language model to use for generation.
            top_k: The number of documents to retrieve.
            memory_key: The key to use for the conversation memory.

        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        self.memory_key = memory_key

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True, output_key="answer")

        # Initialize retrieval chain
        self.chain = self._create_chain()

        logger.info("Initialized ConversationalRAGPipeline")

    def _create_chain(self) -> BaseConversationalRetrievalChain:
        """Create the conversational retrieval chain.

        Returns
        -------
            The conversational retrieval chain.

        """
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k},
        )

        # Limit token usage by truncating document content
        def _format_doc(doc: Document) -> str:
            # Truncate document content to avoid token length issues
            content = doc.page_content
            if len(content) > 150:  # Significantly reduced
                content = content[:150] + "..."
            return content

        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
        )

        return chain

    def query(self, query: str, chat_history: Optional[List[tuple]] = None) -> Dict[str, Any]:
        """Query the RAG pipeline.

        Args:
        ----
            query: The query to answer.
            chat_history: Optional chat history to use instead of the internal
                memory.

        Returns:
        -------
            A dictionary containing the response and retrieved documents.

        """
        logger.info(f"Processing query: {query}")

        try:
            # Limit query length to avoid token issues
            if len(query) > 200:
                query = query[:200]
                logger.warning("Query truncated to 200 characters")

            # Process query
            result = self.chain.invoke({"question": query})

            # Extract response and source documents
            response = result.get("answer", "")
            source_documents = result.get("source_documents", [])

            # Format retrieved documents
            retrieved_documents = []
            for doc in source_documents:
                if isinstance(doc, Document):
                    # Truncate document content if it's too long
                    # (to avoid token length issues)
                    content = doc.page_content
                    if len(content) > 200:
                        content = content[:200] + "..."

                    retrieved_documents.append(
                        {
                            "content": content,
                            "metadata": doc.metadata,
                        }
                    )

            return {
                "response": response,
                "retrieved_documents": retrieved_documents,
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def reset_memory(self) -> None:
        """Reset the conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory reset")
