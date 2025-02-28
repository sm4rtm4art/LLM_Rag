"""API module for the llm-rag package.

This module provides a FastAPI interface for interacting with the RAG system.
"""
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="LLM RAG API",
    description=("API for interacting with the Retrieval-Augmented Generation system"),
    version="0.1.0",
)


class QueryRequest(BaseModel):
    """Query request model."""

    query: str = Field(..., description="The query to process")
    top_k: Optional[int] = Field(3, description="Number of documents to retrieve")


class DocumentMetadata(BaseModel):
    """Document metadata model."""

    source: Optional[str] = Field(None, description="Source of the document")
    filename: Optional[str] = Field(None, description="File name")
    chunk_index: Optional[int] = Field(None, description="Index of the chunk")
    additional_metadata: Optional[Dict] = Field({}, description="Additional metadata")


class Document(BaseModel):
    """Document model."""

    content: str = Field(..., description="The document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")


class QueryResponse(BaseModel):
    """Query response model."""

    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The generated response")
    retrieved_documents: List[Document] = Field(..., description="Retrieved documents used for context")


class ConversationRequest(BaseModel):
    """Conversation request model."""

    query: str = Field(..., description="The query to process")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for tracking history")
    top_k: Optional[int] = Field(3, description="Number of documents to retrieve")


class ConversationTurn(BaseModel):
    """Conversation turn model."""

    user: str = Field(..., description="User message")
    assistant: str = Field(..., description="Assistant response")


class ConversationResponse(QueryResponse):
    """Conversation response model."""

    conversation_id: str = Field(..., description="Conversation ID")
    history: List[ConversationTurn] = Field(..., description="Conversation history")


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint.

    Returns basic API information.

    Returns
    -------
        dict: API information

    """
    return {
        "name": "LLM RAG API",
        "version": "0.1.0",
        "description": "Retrieval-Augmented Generation API",
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Process a single query using the RAG system.

    Args:
    ----
        request: The query request

    Returns:
    -------
        QueryResponse: The response including generated answer
            and retrieved documents

    """
    try:
        # TODO: Implement actual RAG system integration
        # This is a placeholder for demonstration
        mock_response = QueryResponse(
            query=request.query,
            response=f"This is a mock response to: {request.query}",
            retrieved_documents=[
                Document(
                    content="Mock document content",
                    metadata=DocumentMetadata(
                        source="mock_source.txt",
                        filename="mock_source.txt",
                        chunk_index=0,
                        additional_metadata={},
                    ),
                )
            ],
        )
        return mock_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/conversation", response_model=ConversationResponse)
async def conversation(request: ConversationRequest) -> ConversationResponse:
    """Process a query in conversation mode.

    Args:
    ----
        request: The conversation request

    Returns:
    -------
        ConversationResponse: The response including generated answer,
            retrieved documents, and updated conversation history

    """
    try:
        # TODO: Implement conversation handling with RAG
        # This is a placeholder for demonstration
        mock_response = ConversationResponse(
            query=request.query,
            response=f"This is a mock conversational response to: {request.query}",
            retrieved_documents=[
                Document(
                    content="Mock document content for conversation",
                    metadata=DocumentMetadata(
                        source="mock_source.txt",
                        filename="mock_source.txt",
                        chunk_index=0,
                        additional_metadata={},
                    ),
                )
            ],
            conversation_id=request.conversation_id or "new_conversation_123",
            history=[
                ConversationTurn(
                    user=request.query,
                    assistant=f"This is a mock response to: {request.query}",
                )
            ],
        )
        return mock_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
