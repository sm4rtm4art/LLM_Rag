"""Multi-modal RAG Pipeline implementation.

This module contains the implementation of a multi-modal RAG pipeline that can
handle different types of content (text, tables, images) and provide rich responses.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
    ConversationalRetrievalChain,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

from src.llm_rag.vectorstore.multimodal import MultiModalVectorStore

logger = logging.getLogger(__name__)


class MultiModalRAGPipeline:
    """Multi-modal RAG Pipeline.

    This class implements a RAG pipeline that can handle different types of content
    (text, tables, images) and provide rich responses.
    """

    def __init__(
        self,
        vectorstore: Union[VectorStore, MultiModalVectorStore],
        llm: BaseLanguageModel,
        top_k: int = 3,
        memory_key: str = "chat_history",
        content_type_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize the multi-modal RAG pipeline.

        Args:
        ----
            vectorstore: The vector store containing the document embeddings.
            llm: The language model to use for generation.
            top_k: The number of documents to retrieve per content type.
            memory_key: The key to use for the conversation memory.
            content_type_weights: Optional weights for different content types.
                Default is equal weighting.

        """
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        self.memory_key = memory_key

        # Set content type weights (default to equal weighting)
        self.content_type_weights = content_type_weights or {
            "text": 1.0,
            "table": 1.0,
            "image": 1.0,
            "technical_drawing": 1.0,
        }

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True, output_key="answer")

        # Initialize retrieval chain
        self.chain = self._create_chain()

        logger.info("Initialized MultiModalRAGPipeline")

    def _create_chain(self) -> BaseConversationalRetrievalChain:
        """Create the conversational retrieval chain.

        Returns
        -------
            The conversational retrieval chain.

        """
        # Create retriever
        if isinstance(self.vectorstore, MultiModalVectorStore):
            # Use specialized multi-modal retriever
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        else:
            # Use standard retriever
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        # Create custom prompt that handles multi-modal content
        prompt_template = """
        You are an assistant that answers questions based on multiple sources of information,
        including text, tables, and images.

        Chat History:
        {chat_history}

        Question: {question}

        I'll provide you with relevant information from different sources:

        TEXT SOURCES:
        {text_sources}

        TABLE SOURCES:
        {table_sources}

        IMAGE SOURCES:
        {image_sources}

        Based on the above information, please provide a comprehensive answer.
        If referring to images, mention them specifically (e.g., "As shown in Figure 1...").
        If using data from tables, cite the specific table.
        If you don't know the answer, say so - don't make up information.

        Answer:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["chat_history", "question", "text_sources", "table_sources", "image_sources"],
        )

        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
        )

        return chain

    def _format_documents_by_type(self, documents: List[Document]) -> Dict[str, str]:
        """Format documents by content type for the prompt.

        Args:
        ----
            documents: List of retrieved documents.

        Returns:
        -------
            Dictionary with formatted content for each type.

        """
        # Organize documents by content type
        docs_by_type = {
            "text": [],
            "table": [],
            "image": [],
            "technical_drawing": [],
        }

        for doc in documents:
            filetype = doc.metadata.get("filetype", "")

            if filetype in ["image", "technical_drawing"]:
                docs_by_type["image"].append(doc)
            elif filetype == "table":
                docs_by_type["table"].append(doc)
            else:
                docs_by_type["text"].append(doc)

        # Format each content type
        formatted_content = {}

        # Format text documents
        if docs_by_type["text"]:
            text_content = []
            for i, doc in enumerate(docs_by_type["text"]):
                source = doc.metadata.get("source", "Unknown")
                section = doc.metadata.get("section_title", "")
                text_content.append(f"[Text {i + 1}] {section} (Source: {source})\n{doc.page_content}\n")
            formatted_content["text_sources"] = "\n".join(text_content)
        else:
            formatted_content["text_sources"] = "No relevant text sources found."

        # Format table documents
        if docs_by_type["table"]:
            table_content = []
            for i, doc in enumerate(docs_by_type["table"]):
                source = doc.metadata.get("source", "Unknown")
                table_index = doc.metadata.get("table_index", i)
                table_content.append(f"[Table {table_index + 1}] (Source: {source})\n{doc.page_content}\n")
            formatted_content["table_sources"] = "\n".join(table_content)
        else:
            formatted_content["table_sources"] = "No relevant table sources found."

        # Format image documents
        if docs_by_type["image"]:
            image_content = []
            for i, doc in enumerate(docs_by_type["image"]):
                source = doc.metadata.get("source", "Unknown")
                page_num = doc.metadata.get("page_num", "Unknown")
                image_path = doc.metadata.get("image_path", "")
                image_type = "Technical Drawing" if doc.metadata.get("filetype") == "technical_drawing" else "Figure"

                description = f"[{image_type} {i + 1}] From page {page_num} (Source: {source})\n"
                description += f"Description: {doc.page_content}\n"
                description += f"Image path: {image_path}\n"

                image_content.append(description)
            formatted_content["image_sources"] = "\n".join(image_content)
        else:
            formatted_content["image_sources"] = "No relevant image sources found."

        return formatted_content

    def query(self, query: str, chat_history: Optional[List[tuple]] = None) -> Dict[str, Any]:
        """Query the multi-modal RAG pipeline.

        Args:
        ----
            query: The query to answer.
            chat_history: Optional chat history to use instead of the internal memory.

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

            # Get documents from the retriever
            if isinstance(self.vectorstore, MultiModalVectorStore):
                # Use multi-modal retriever to get documents by content type
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
                documents = retriever.get_relevant_documents(query)
            else:
                # Use standard retriever
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
                documents = retriever.get_relevant_documents(query)

            # Format documents by content type
            formatted_content = self._format_documents_by_type(documents)

            # Process query with the chain
            result = self.chain.invoke(
                {
                    "question": query,
                    **formatted_content,
                }
            )

            # Extract response and source documents
            response = result.get("answer", "")
            source_documents = result.get("source_documents", [])

            # Format retrieved documents
            retrieved_documents = []
            for doc in source_documents:
                if isinstance(doc, Document):
                    # Truncate document content if it's too long
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
