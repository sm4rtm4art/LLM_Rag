#!/bin/bash
set -e

echo "=== Testing Dockerfile ==="
echo "Building Docker image..."
docker build -t llm-rag:test .

echo -e "\n=== Testing CLI mode ==="
echo "Running container in CLI mode with --help..."
docker run --rm llm-rag:test --help

echo -e "\n=== Testing API mode ==="
echo "Running container in API mode (will be terminated after 5 seconds)..."
docker run --rm -d -p 8000:8000 --name llm-rag-api-test llm-rag:test api
sleep 5

echo "Testing API health endpoint..."
curl -s http://localhost:8000/health || echo "API health check failed"

echo "Stopping API container..."
docker stop llm-rag-api-test

echo -e "\n=== Docker tests completed ==="
