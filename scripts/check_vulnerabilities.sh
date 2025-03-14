#!/bin/bash
# Script to check for Python package vulnerabilities using Safety

set -e

# Check if API key is available as environment variable
if [ -n "${SAFETY_API_KEY}" ]; then
    echo "Using Safety with API key from environment variable"
    SAFETY_CMD="safety scan --key ${SAFETY_API_KEY}"
else
    # Otherwise use authenticated method (safety auth)
    echo "Using Safety with authentication from safety auth"
    SAFETY_CMD="safety scan"
fi

# Run safety check on dependencies
echo "Checking for vulnerabilities in Python dependencies..."
$SAFETY_CMD

# If safety check passes, exit with success
echo "Vulnerability check completed."
