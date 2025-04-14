#!/bin/bash
# Run tests in CI mode, which will skip tests that require access to real PDF files

# Set the CI environment variable to true
export CI=true

# Run the specified tests or all tests if none are specified
if [ $# -eq 0 ]; then
    echo "Running all tests in CI mode (skipping tests that require real PDF files)"
    python -m pytest
else
    echo "Running specified tests in CI mode (skipping tests that require real PDF files)"
    python -m pytest "$@"
fi

# Unset the CI environment variable
unset CI
