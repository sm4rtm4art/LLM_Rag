#!/bin/bash
# Script to run RAG evaluation tests

# Usage information
usage() {
    echo "Usage: $0"
    echo "Runs the RAG evaluation tests for the project."
    echo
    echo "This script executes the pytest tests in tests/test_rag_evaluation.py"
    echo "with verbose output to validate the RAG system functionality."
    echo
    echo "No arguments are required."
    exit 1
}

# Show usage if help flag is provided
if [[ $1 == "-h" || $1 == "--help" ]]; then
    usage
fi

set -e # Exit on error

# Change to the project root directory
cd "$(dirname "$0")/.."

# Run the tests
echo "Running RAG evaluation tests..."
if pytest tests/test_rag_evaluation.py -v; then
    echo "✅ All RAG evaluation tests passed!"
else
    echo "❌ Some RAG evaluation tests failed."
    exit 1
fi
