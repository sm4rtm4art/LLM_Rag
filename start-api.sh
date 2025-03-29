#!/bin/bash

# Build and run the LLM RAG API container
echo "Building LLM RAG API Docker image..."
docker build -t llm-rag:latest .

echo "Starting LLM RAG API..."
docker-compose -f docker-compose.api.yml up -d

echo "API running at http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
