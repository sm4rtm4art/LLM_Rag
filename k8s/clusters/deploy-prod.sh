#!/bin/bash
set -e

# Production deployment script for LLM RAG
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
KUSTOMIZE_DIR="${BASE_DIR}/k8s/overlays/production"

# Configuration variables
NAMESPACE=${NAMESPACE:-llm-rag-prod}
KUBECONFIG=${KUBECONFIG:-~/.kube/config}
CONTEXT=${CONTEXT:-production}

# Print header
echo "🚀 Deploying LLM RAG to production environment"
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

# Switch to the correct context
kubectl config use-context "$CONTEXT"

# Create namespace if it doesn't exist
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "📁 Creating namespace $NAMESPACE..."
    kubectl create namespace "$NAMESPACE"
fi

# Apply kustomize configuration
echo "🔧 Applying kustomize configuration..."
kustomize build "$KUSTOMIZE_DIR" | kubectl apply -f -

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl -n "$NAMESPACE" rollout status deployment/llm-rag --timeout=300s

# Verify the deployment
echo "✅ Deployment completed successfully!"
echo ""
echo "📊 Deployment status:"
kubectl -n "$NAMESPACE" get deployments,pods,svc,ingress

echo ""
echo "🔗 Your application should be available at: https://api.llm-rag.com"
echo ""
echo "To check the logs:"
echo "kubectl -n $NAMESPACE logs -l app=llm-rag -f"
