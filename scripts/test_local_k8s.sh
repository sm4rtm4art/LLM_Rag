#!/bin/bash
set -e

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Testing local Kubernetes deployment...${NC}"

# Check if KIND is installed
if ! command -v kind &>/dev/null; then
    echo -e "${RED}KIND is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if the cluster exists
if ! kind get clusters | grep -q "llm-rag-local"; then
    echo -e "${RED}Cluster 'llm-rag-local' does not exist. Please run setup_local_k8s.sh first.${NC}"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &>/dev/null; then
    echo -e "${RED}kubectl is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if the deployment exists
if ! kubectl get deployment llm-rag -n llm-rag-test &>/dev/null; then
    echo -e "${RED}Deployment 'llm-rag' does not exist in namespace 'llm-rag-test'.${NC}"
    echo -e "${RED}Please run setup_local_k8s.sh first.${NC}"
    exit 1
fi

# Start port forwarding in the background
echo -e "${YELLOW}Starting port forwarding...${NC}"
kubectl port-forward svc/llm-rag -n llm-rag-test 8000:8000 &
PORT_FORWARD_PID=$!

# Make sure to kill the port forwarding when the script exits
trap 'kill $PORT_FORWARD_PID 2>/dev/null || true' EXIT

# Wait for port forwarding to be established
echo -e "${YELLOW}Waiting for port forwarding to be established...${NC}"
sleep 3

# Test the health endpoint
echo -e "${YELLOW}Testing health endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health || echo "Failed to connect")

if [[ $HEALTH_RESPONSE == *"status"* ]]; then
    echo -e "${GREEN}Health endpoint is working!${NC}"
    echo -e "Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}Health endpoint is not working.${NC}"
    echo -e "Response: $HEALTH_RESPONSE"
    exit 1
fi

# Test a simple query
echo -e "${YELLOW}Testing a simple query...${NC}"
QUERY_RESPONSE=$(curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What is RAG?", "top_k": 1}' || echo "Failed to connect")

if [[ $QUERY_RESPONSE == *"response"* ]]; then
    echo -e "${GREEN}Query endpoint is working!${NC}"
    echo -e "Response: $QUERY_RESPONSE"
else
    echo -e "${RED}Query endpoint is not working.${NC}"
    echo -e "Response: $QUERY_RESPONSE"
    exit 1
fi

echo -e "${GREEN}All tests passed!${NC}"
