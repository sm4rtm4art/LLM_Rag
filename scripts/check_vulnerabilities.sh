#!/bin/bash
# Script to check for Python package vulnerabilities using Safety

set -e

# Display usage information
usage() {
    echo "Usage: $(basename "$0") [options]"
    echo
    echo "Check Python dependencies for security vulnerabilities using Safety."
    echo
    echo "Options:"
    echo "  -h, --help     Display this help message and exit"
    echo "  --key KEY      Use the specified Safety API key instead of authenticated method"
    echo
    echo "Environment variables:"
    echo "  SAFETY_API_KEY  If set, will be used as the Safety API key"
    echo
    echo "Examples:"
    echo "  $(basename "$0")                   # Run with authenticated method"
    echo "  $(basename "$0") --key YOUR_KEY    # Run with specified API key"
    echo "  SAFETY_API_KEY=YOUR_KEY $(basename "$0")  # Run with API key from environment"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --key)
            if [[ -n "$2" ]]; then
                SAFETY_API_KEY="$2"
                shift 2
            else
                echo "Error: --key requires an argument" >&2
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

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
