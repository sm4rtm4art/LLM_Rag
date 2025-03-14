#!/bin/bash
set -e

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display usage information
usage() {
    echo "Usage: $(basename "$0")"
    echo
    echo "Clean up a local Kubernetes cluster created with KIND."
    echo "This script deletes the 'llm-rag-local' cluster to free up resources."
    echo
    echo "Prerequisites:"
    echo "  - KIND (Kubernetes IN Docker) must be installed"
    echo
    echo "No options or arguments are required."
}

# Parse command line arguments
if [[ $1 == "-h" || $1 == "--help" ]]; then
    usage
    exit 0
fi

echo -e "${YELLOW}Cleaning up local Kubernetes cluster...${NC}"

# Check if KIND is installed
if ! command -v kind &>/dev/null; then
    echo -e "${RED}KIND is not installed. Nothing to clean up.${NC}"
    exit 1
fi

# Check if the cluster exists
if ! kind get clusters | grep -q "llm-rag-local"; then
    echo -e "${YELLOW}Cluster 'llm-rag-local' does not exist. Nothing to clean up.${NC}"
    exit 0
fi

# Delete the cluster
echo -e "${YELLOW}Deleting KIND cluster 'llm-rag-local'...${NC}"
kind delete cluster --name llm-rag-local

echo -e "${GREEN}Cleanup complete!${NC}"
