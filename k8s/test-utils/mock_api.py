from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title='LLM-RAG Mock API', description='A mock API for Kubernetes testing', version='0.1.0')


class QueryRequest(BaseModel):
    """Model representing a query request to the API.

    Contains the query string, top_k parameter for limiting results,
    and optional filters.
    """

    query: str
    top_k: Optional[int] = 1
    filters: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Model representing a query response from the API.

    Contains the response text and optional sources/citations.
    """

    response: str
    sources: Optional[list] = []


@app.get('/health')
async def health_check():
    """Health check endpoint for Kubernetes liveness and readiness probes."""
    return {'status': 'healthy'}


@app.post('/query', response_model=QueryResponse)
async def query(request: QueryRequest):
    """Mock query endpoint that responds to any query."""
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail='Query cannot be empty')

    # Return a simple mock response for any query
    return {
        'response': f"This is a mock response to: '{request.query}'",
        'sources': [{'title': 'Mock Document', 'url': 'https://example.com/doc1', 'relevance': 0.95}],
    }


@app.get('/')
async def root():
    """Root endpoint with basic information."""
    return {
        'name': 'LLM-RAG Mock API',
        'version': '0.1.0',
        'description': 'Mock API for Kubernetes testing',
        'endpoints': ['/health', '/query'],
    }
