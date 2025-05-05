#!/bin/bash
# Script to test Kubernetes configurations

# Exit on error, but ensure cleanup happens
set -e
trap cleanup EXIT

# Cleanup function to ensure we always remove test clusters
cleanup() {
    echo "Cleaning up resources..."
    if kind get clusters 2>/dev/null | grep -q "temp-test"; then
        echo "Removing test cluster..."
        kind delete cluster --name temp-test
    fi
}

# Usage information
usage() {
    echo "Usage: $0 [--no-cluster]"
    echo "Tests Kubernetes configuration files in the k8s directory."
    echo
    echo "Options:"
    echo "  --no-cluster  Skip the kind cluster creation test"
    echo
    echo "This script performs the following checks:"
    echo "  1. Validates Kubernetes YAML files using kubectl dry-run"
    echo "  2. Validates files with kubeval (if installed)"
    echo "  3. Tests kind cluster creation (if kind is installed and --no-cluster not specified)"
    exit 1
}

# Parse arguments
SKIP_CLUSTER=false
if [[ $1 == "--no-cluster" ]]; then
    SKIP_CLUSTER=true
fi

# Show usage if help flag is provided
if [[ $1 == "-h" || $1 == "--help" ]]; then
    usage
fi

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

# First, check that required files exist
echo -e "\n=== Checking for required files ==="
REQUIRED_FILES=("deployment.yaml" "service.yaml" "configmap.yaml" "kind-config.yaml")
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${K8S_DIR}/${file}" ]]; then
        echo "ERROR: Required file ${K8S_DIR}/${file} is missing!"
        exit 1
    else
        echo "✓ ${K8S_DIR}/${file} exists"
    fi
done

# Validate using kubectl dry-run
echo -e "\n=== Validating with kubectl dry-run ==="
for file in "${K8S_DIR}"/*.yaml; do
    # Skip non-Kubernetes manifests
    if [[ -f $file && $file != *"kind-config.yaml"* && $file != *"test-script.sh"* && $file != *"kustomization.yaml"* ]]; then
        echo "Validating $file..."
        if ! kubectl apply --dry-run=client -f "$file"; then
            echo "ERROR: $file failed validation"
            exit 1
        fi
    fi
done

# Validate using kubeval if available
if [ "$KUBEVAL_AVAILABLE" = true ]; then
    echo -e "\n=== Validating with kubeval ==="
    for file in "${K8S_DIR}"/*.yaml; do
        # Skip non-Kubernetes manifests
        if [[ -f $file && $file != *"kind-config.yaml"* && $file != *"test-script.sh"* && $file != *"kustomization.yaml"* ]]; then
            echo "Validating $file with kubeval..."
            if ! kubeval "$file"; then
                echo "WARNING: $file failed kubeval validation"
                # Don't exit here as kubeval can be stricter than kubectl
            fi
        fi
    done
fi

# Test kind cluster creation (if kind is installed and not skipped)
if command -v kind &>/dev/null && [ "$SKIP_CLUSTER" = false ]; then
    echo -e "\n=== Testing kind cluster configuration ==="
    echo "Validating kind-config.yaml..."

    # Check if the cluster already exists
    if kind get clusters 2>/dev/null | grep -q "temp-test"; then
        echo "Cluster temp-test already exists, deleting it first..."
        kind delete cluster --name temp-test
    fi

    # Create the cluster with a timeout
    echo "Creating test cluster with 60s timeout..."
    timeout 60s kind create cluster --name temp-test --config "${K8S_DIR}/kind-config.yaml" --wait 30s

    # Check if cluster was created successfully
    if kind get clusters | grep -q "temp-test"; then
        echo "✓ Cluster created successfully"
    else
        echo "ERROR: Failed to create test cluster"
        exit 1
    fi
else
    if [ "$SKIP_CLUSTER" = true ]; then
        echo -e "\n=== Skipping cluster creation test (--no-cluster specified) ==="
    else
        echo -e "\n=== kind is not installed, skipping cluster creation test ==="
        echo "To install kind:"
        echo "  brew install kind (macOS)"
        echo "  or visit https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
    fi
fi

echo -e "\n=== Kubernetes validation completed successfully! ==="
