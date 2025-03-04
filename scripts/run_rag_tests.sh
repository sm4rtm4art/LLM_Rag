#!/bin/bash
# Script to run RAG evaluation tests

set -e  # Exit on error

# Change to the project root directory
cd "$(dirname "$0")/.."

# Run the tests
echo "Running RAG evaluation tests..."
pytest tests/test_rag_evaluation.py -v

# If tests pass, show success message
if [ $? -eq 0 ]; then
    echo "✅ All RAG evaluation tests passed!"
else
    echo "❌ Some RAG evaluation tests failed."
    exit 1
fi
