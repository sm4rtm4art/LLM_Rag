#!/bin/bash
# Simple wrapper script for running tests directly with pytest

# Set the directory to the project root
cd "$(dirname "$0")/../.." || exit 1

# Parse arguments
COVERAGE=""
VERBOSE=""
for arg in "$@"; do
    case $arg in
        -c | --coverage)
            COVERAGE="--cov=src/llm_rag --cov-report=xml"
            ;;
        -v | --verbose)
            VERBOSE="-v"
            ;;
    esac
done

# Run pytest directly with appropriate flags
echo "Running pytest $VERBOSE $COVERAGE tests"
pytest $VERBOSE $COVERAGE tests
