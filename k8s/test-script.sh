#!/bin/bash
set -e

# Wait for the service to be ready
echo "Waiting for LLM RAG service to be ready..."
timeout=60
counter=0
ENDPOINT="http://llm-rag/health"

until $(curl --output /dev/null --silent --fail $ENDPOINT); do
    if [ $counter -eq $timeout ]; then
        echo "Timed out waiting for service to be ready"
        exit 1
    fi
    counter=$((counter + 1))
    echo "Waiting for service to be ready... ($counter/$timeout)"
    sleep 1
done

echo "Service is ready!"

# Test health endpoint
health_response=$(curl -s http://llm-rag/health)
echo "Health endpoint response: $health_response"

# Test the API by making a simple query
echo "Testing API with a simple query..."
query_response=$(curl -s -X POST http://llm-rag/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What is RAG?", "top_k": 1}')

echo "Query response: $query_response"

# Check if the response is valid JSON
if echo "$query_response" | grep -q "response"; then
    echo "✅ Test passed! API is working correctly"
    exit 0
else
    echo "❌ Test failed! API response is not as expected"
    exit 1
fi
