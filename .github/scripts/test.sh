#!/bin/bash
# Simple wrapper script for running tests with the Rich progress display

# Set the directory to the project root
cd "$(dirname "$0")/../.." || exit 1

# Run the test script with all arguments passed to this script
python .github/scripts/run_tests.py "$@"

