#!/bin/bash
set -e

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up local Kubernetes cluster with KIND...${NC}"

# Check if KIND is installed
if ! command -v kind &>/dev/null; then
    echo -e "${RED}KIND is not installed. Please install it first:${NC}"
    echo "https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &>/dev/null; then
    echo -e "${RED}kubectl is not installed. Please install it first:${NC}"
    echo "https://kubernetes.io/docs/tasks/tools/install-kubectl/"
    exit 1
fi

# Check if Docker is running
if ! docker info &>/dev/null; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Create KIND cluster using our configuration
echo -e "${YELLOW}Creating KIND cluster...${NC}"
kind create cluster --name llm-rag-local --config k8s/kind-config.yaml

# Wait for the cluster to be ready
echo -e "${YELLOW}Waiting for the cluster to be ready...${NC}"
kubectl wait --for=condition=Ready nodes --all --timeout=60s

# Create namespace
echo -e "${YELLOW}Creating namespace...${NC}"
kubectl create namespace llm-rag-test || true

# Create ConfigMap
echo -e "${YELLOW}Creating ConfigMap...${NC}"
kubectl apply -f k8s/configmap.yaml -n llm-rag-test

# Create PVC - using local-pvc.yaml for KIND
echo -e "${YELLOW}Creating PVC...${NC}"
kubectl apply -f k8s/local-pvc.yaml -n llm-rag-test

# Create a local hostPath PV for the PVC to bind to
echo -e "${YELLOW}Creating local PV...${NC}"
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: models-pv
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  hostPath:
    path: /tmp/models-data
    type: DirectoryOrCreate
EOF

# Build and load the Docker image into KIND
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t llm-rag:local .

echo -e "${YELLOW}Loading Docker image into KIND...${NC}"
kind load docker-image llm-rag:local --name llm-rag-local

# Deploy using the local deployment configuration
echo -e "${YELLOW}Deploying to Kubernetes...${NC}"
kubectl apply -f k8s/local-deployment.yaml -n llm-rag-test

# Create service using the local service configuration
echo -e "${YELLOW}Creating service...${NC}"
kubectl apply -f k8s/local-service.yaml -n llm-rag-test

# Wait for deployment to be ready
echo -e "${YELLOW}Waiting for deployment to be ready...${NC}"
kubectl rollout status deployment/llm-rag -n llm-rag-test --timeout=120s

# Show pods
echo -e "${YELLOW}Pods:${NC}"
kubectl get pods -n llm-rag-test

# Show services
echo -e "${YELLOW}Services:${NC}"
kubectl get services -n llm-rag-test

# Port forward to access the service locally
echo -e "${GREEN}Setting up port forwarding from localhost:8000 to the service...${NC}"
echo -e "${GREEN}Press Ctrl+C to stop port forwarding when done testing${NC}"
kubectl port-forward svc/llm-rag -n llm-rag-test 8000:8000

# Note: The script will stop here while port forwarding is active
# After Ctrl+C, the script will continue

echo -e "${YELLOW}Cleaning up...${NC}"
echo -e "${YELLOW}To delete the cluster, run: kind delete cluster --name llm-rag-local${NC}"
echo -e "${GREEN}Done!${NC}"
