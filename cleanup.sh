#!/bin/bash
set -e
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -not -path "./.llm_rag/*" -exec rm -rf {} +
echo "Removing .coverage files..."
find . -name ".coverage*" -not -path "./.llm_rag/*" -delete
echo "Removing .egg-info directories..."
find . -type d -name "*.egg-info" -exec rm -rf {} +
echo "Removing unnecessary database directories..."
rm -rf --db-path/
echo "Clean up complete!"
