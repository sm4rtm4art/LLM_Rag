#!/bin/bash
# Script to run promptfoo tests for the LLM RAG system

set -e # Exit on error

# Check if we're in the right directory
if [ ! -f "promptfoo.yaml" ]; then
    echo "Error: promptfoo.yaml not found. Please run this script from the tests/promptfoo directory."
    exit 1
fi

# Check if promptfoo is installed
if ! command -v promptfoo &>/dev/null; then
    echo "promptfoo not found. Installing..."
    cd ../../
    uv pip install -e ".[dev]"
    cd tests/promptfoo
fi

# Run the tests
echo "Running promptfoo tests..."
promptfoo eval

# Check if the results were generated
if [ -f "results.html" ]; then
    echo "Tests completed successfully. Results available in:"
    echo "- $(pwd)/results.json"
    echo "- $(pwd)/results.html"

    # Open the results in the browser if on macOS
    if [[ $OSTYPE == "darwin"* ]]; then
        echo "Opening results in browser..."
        open results.html
    fi
else
    echo "Error: Test results not generated."
    exit 1
fi
