#!/bin/bash
set -e

# Development deployment script for LLM RAG
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
KUSTOMIZE_DIR="${BASE_DIR}/k8s/overlays/dev"

# Configuration variables
NAMESPACE=${NAMESPACE:-llm-rag-dev}
KUBECONFIG=${KUBECONFIG:-~/.kube/config}
CONTEXT=${CONTEXT:-kind-llm-rag}

# Print header
echo "🧪 Deploying LLM RAG to development environment"
echo "==============================================="
echo "Namespace: $NAMESPACE"
echo "Context: $CONTEXT"
echo "Using kustomize directory: $KUSTOMIZE_DIR"
echo ""

# Check if required tools are installed
for cmd in kubectl kustomize; do
    if ! command -v $cmd &>/dev/null; then
        echo "❌ $cmd is not installed. Please install it first."
        exit 1
    fi
done

# Check if KIND cluster is running, create if not
if ! kind get clusters | grep -q "$CONTEXT"; then
    echo "🔄 Creating KIND cluster..."
    kind create cluster --name llm-rag --config "${BASE_DIR}/k8s/kind-config.yaml"
fi

# Switch to the correct context
kubectl config use-context "$CONTEXT"

# Create namespace if it doesn't exist
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "📁 Creating namespace $NAMESPACE..."
    kubectl create namespace "$NAMESPACE"
fi

# Build and load the Docker image if needed
if [[ $REBUILD_IMAGE == "true" ]]; then
    echo "🔨 Building Docker image..."
    docker build -t llm-rag:latest "${BASE_DIR}"

    echo "📦 Loading image into KIND cluster..."
    kind load docker-image llm-rag:latest --name llm-rag
fi

# Apply kustomize configuration
echo "🔧 Applying kustomize configuration..."
kustomize build "$KUSTOMIZE_DIR" | kubectl apply -f -

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl -n "$NAMESPACE" rollout status deployment/llm-rag --timeout=180s

# Set up port forwarding for local access
echo "🔌 Setting up port forwarding..."
kubectl -n "$NAMESPACE" port-forward svc/llm-rag 8000:80 &
PORT_FORWARD_PID=$!

# Verify the deployment
echo "✅ Deployment completed successfully!"
echo ""
echo "📊 Deployment status:"
kubectl -n "$NAMESPACE" get deployments,pods,svc,ingress

echo ""
echo "🔗 Your application should be available at: http://localhost:8000"
echo "🔗 Or via ingress at: http://dev.llm-rag.local"
echo ""
echo "Add this to your /etc/hosts to use the ingress hostname:"
echo "127.0.0.1 dev.llm-rag.local"
echo ""
echo "Press Ctrl+C to stop port forwarding"

# Trap to clean up port forwarding
trap 'kill $PORT_FORWARD_PID 2>/dev/null || true' EXIT

# Wait for user to terminate with Ctrl+C
wait $PORT_FORWARD_PID
