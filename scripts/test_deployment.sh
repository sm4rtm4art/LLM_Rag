#!/bin/bash
set -e

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${GREEN}=== $1 ===${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Parse arguments
TEST_DOCKER=false
TEST_K8S=false
TEST_ALL=false

if [ $# -eq 0 ]; then
    TEST_ALL=true
else
    for arg in "$@"; do
        case $arg in
            --docker)
                TEST_DOCKER=true
                ;;
            --k8s)
                TEST_K8S=true
                ;;
            --all)
                TEST_ALL=true
                ;;
            --help)
                echo "Usage: $0 [--docker] [--k8s] [--all]"
                echo "  --docker: Test Docker configuration"
                echo "  --k8s: Test Kubernetes configuration"
                echo "  --all: Test both Docker and Kubernetes (default if no args provided)"
                exit 0
                ;;
            *)
                print_error "Unknown argument: $arg"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
fi

if [ "$TEST_ALL" = true ]; then
    TEST_DOCKER=true
    TEST_K8S=true
fi

# Check script dependencies
if [ "$TEST_DOCKER" = true ] && ! command -v docker &>/dev/null; then
    print_error "Docker is not installed. Please install it first."
    exit 1
fi

if [ "$TEST_K8S" = true ] && ! command -v kubectl &>/dev/null; then
    print_error "kubectl is not installed. Please install it first."
    exit 1
fi

# Run Docker tests
if [ "$TEST_DOCKER" = true ]; then
    print_header "Testing Docker Configuration"
    ./scripts/test_docker.sh
fi

# Run Kubernetes tests
if [ "$TEST_K8S" = true ]; then
    print_header "Testing Kubernetes Configuration"
    ./scripts/test_kubernetes.sh
fi

print_header "All tests completed"
