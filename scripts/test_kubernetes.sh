#!/bin/bash
# Script to test Kubernetes configurations

# Usage information
usage() {
    echo "Usage: $0"
    echo "Tests Kubernetes configuration files in the k8s directory."
    echo
    echo "This script performs the following checks:"
    echo "  1. Validates Kubernetes YAML files using kubectl dry-run"
    echo "  2. Validates files with kubeval (if installed)"
    echo "  3. Tests kind cluster creation (if kind is installed)"
    echo
    echo "No arguments are required."
    exit 1
}

# Show usage if help flag is provided
if [[ $1 == "-h" || $1 == "--help" ]]; then
    usage
fi

set -e

echo "=== Testing Kubernetes Configurations ==="

# Check if kubectl is installed
if ! command -v kubectl &>/dev/null; then
    echo "kubectl is not installed. Please install it first."
    exit 1
fi

# Check if kubeval is installed (optional)
KUBEVAL_AVAILABLE=false
if command -v kubeval &>/dev/null; then
    KUBEVAL_AVAILABLE=true
    echo "kubeval is available, will use it for validation"
else
    echo "kubeval is not installed. Will use kubectl dry-run only."
    echo "For more thorough validation, consider installing kubeval:"
    echo "  brew install kubeval (macOS)"
    echo "  or visit https://github.com/instrumenta/kubeval"
fi

# Directory containing Kubernetes manifests
K8S_DIR="k8s"

# Validate using kubectl dry-run
echo -e "\n=== Validating with kubectl dry-run ==="
for file in "${K8S_DIR}"/*.yaml; do
    if [[ -f $file && $file != *"kind-config.yaml"* && $file != *"test-script.sh"* ]]; then
        echo "Validating $file..."
        kubectl apply --dry-run=client -f "$file"
    fi
done

# Validate using kubeval if available
if [ "$KUBEVAL_AVAILABLE" = true ]; then
    echo -e "\n=== Validating with kubeval ==="
    for file in "${K8S_DIR}"/*.yaml; do
        if [[ -f $file && $file != *"kind-config.yaml"* && $file != *"test-script.sh"* ]]; then
            echo "Validating $file with kubeval..."
            kubeval "$file"
        fi
    done
fi

# Test kind cluster creation (if kind is installed)
if command -v kind &>/dev/null; then
    echo -e "\n=== Testing kind cluster configuration ==="
    echo "Validating kind-config.yaml..."
    kind get kubeconfig --name temp-test 2>/dev/null || kind create cluster --name temp-test --config "${K8S_DIR}/kind-config.yaml" --wait 30s

    echo "Cleaning up test cluster..."
    kind delete cluster --name temp-test
else
    echo -e "\n=== kind is not installed, skipping cluster creation test ==="
    echo "To install kind:"
    echo "  brew install kind (macOS)"
    echo "  or visit https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
fi

echo -e "\n=== Kubernetes validation completed ==="
