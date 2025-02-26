"""FastAPI application for LLM RAG service."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLM RAG API", version="0.1.0")


class QueryRequest(BaseModel):
    """Query request model."""

    query: str
    context: str | None = None


@app.post("/query")
async def process_query(request: QueryRequest) -> dict:
    """Process a RAG query."""
    return {"result": "Placeholder for RAG response"}
