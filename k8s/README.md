# Kubernetes Deployment for LLM-RAG

[![Kubernetes Tests](https://github.com/sm4rtm4art/LLM_Rag/actions/workflows/k8s-test.yaml/badge.svg)](https://github.com/sm4rtm4art/LLM_Rag/actions/workflows/k8s-test.yaml)

This directory contains Kubernetes configuration files for deploying the LLM-RAG application.

## Local Development with KIND

We've provided scripts to easily set up and test a local Kubernetes cluster using KIND (Kubernetes IN Docker).

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
- [KIND](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)

### Scripts

We provide three scripts to help you work with the local Kubernetes cluster:

1. **Setup Script**: `scripts/setup_local_k8s.sh`

   - Creates a KIND cluster
   - Builds and loads the Docker image
   - Deploys the application
   - Sets up port forwarding

2. **Test Script**: `scripts/test_local_k8s.sh`

   - Tests the health endpoint
   - Tests a simple query
   - Verifies the deployment is working

3. **Cleanup Script**: `scripts/cleanup_local_k8s.sh`
   - Deletes the KIND cluster
   - Cleans up all resources

### Usage

```bash
# Set up the local Kubernetes cluster and deploy the application
./scripts/setup_local_k8s.sh

# Test the deployment
./scripts/test_local_k8s.sh

# Clean up when you're done
./scripts/cleanup_local_k8s.sh
```

## Manual Deployment

You can also manually deploy the application to a Kubernetes cluster:

```bash
# Create a KIND cluster
kind create cluster --name llm-rag-cluster --config k8s/kind-config.yaml

# Create a namespace
kubectl create namespace llm-rag-dev

# Apply the Kubernetes configurations
kubectl apply -f k8s/configmap.yaml -n llm-rag-dev
kubectl apply -f k8s/pvc.yaml -n llm-rag-dev
kubectl apply -f k8s/deployment.yaml -n llm-rag-dev
kubectl apply -f k8s/service.yaml -n llm-rag-dev
kubectl apply -f k8s/ingress.yaml -n llm-rag-dev

# Verify the deployment
kubectl get pods -n llm-rag-dev
kubectl get services -n llm-rag-dev
```

## Configuration Files

- `deployment.yaml`: Defines the Kubernetes Deployment
- `service.yaml`: Defines the Kubernetes Service
- `configmap.yaml`: Contains configuration data
- `pvc.yaml`: Defines the Persistent Volume Claim for models
- `ingress.yaml`: Defines the Ingress for external access
- `hpa.yaml`: Horizontal Pod Autoscaler for scaling
- `network-policy.yaml`: Network policies for security
- `kind-config.yaml`: Configuration for the KIND cluster

## CI/CD Integration

The CI/CD pipeline uses these configuration files to deploy the application to test, development, and staging environments.

### Kubernetes Testing Workflow

The `k8s-test.yaml` workflow in GitHub Actions tests the Kubernetes deployment:

1. **Setup**: Creates a KIND cluster with the configuration in `kind-config.yaml`
2. **Build or Pull**: Either builds a new Docker image or pulls the image built by the CI/CD pipeline
3. **Deploy**: Deploys the application to the KIND cluster
4. **Test**: Runs tests against the deployed application
5. **Cleanup**: Deletes the KIND cluster and cleans up resources

This workflow is configured as non-blocking, so failures won't prevent merges to the main branch.

### Deployment Workflow

The main CI/CD pipeline (`ci-cd.yml`) includes deployment jobs that:

1. **Build and Push**: Build the Docker image and push it to Docker Hub
2. **Deploy to Dev**: Deploy the application to the development environment
3. **Deploy to Staging**: Deploy the application to the staging environment (on manual trigger)

The deployment jobs use either:

- A KIND cluster created during the workflow
- An external Kubernetes cluster configured via GitHub Secrets

### Environment-Specific Configurations

The deployment can be customized for different environments:

- **Development**: Uses the `llm-rag-dev` namespace
- **Staging**: Uses the `llm-rag-staging` namespace
- **Production**: (Not yet implemented)

## Troubleshooting

If you encounter issues with the local Kubernetes deployment:

1. Check if Docker is running
2. Verify that KIND and kubectl are installed
3. Check the logs of the pods:
   ```bash
   kubectl logs -l app=llm-rag -n llm-rag-test
   ```
4. Describe the pods to see any issues:
   ```bash
   kubectl describe pods -l app=llm-rag -n llm-rag-test
   ```
5. Check the events:
   ```bash
   kubectl get events -n llm-rag-test
   ```
6. Verify your kubectl context:
   ```bash
   kubectl config current-context
   ```
7. Check if the KIND cluster is running:
   ```bash
   kind get clusters
   ```

## Common Issues and Solutions

### No Space Left on Device

If you encounter "No space left on device" errors in GitHub Actions:

1. Use the free disk space step in the workflow
2. Optimize your Docker image size
3. Clean up Docker resources after use

### Connection Refused

If you see "connection refused" errors when trying to connect to the Kubernetes API:

1. Make sure the KIND cluster is running
2. Check your kubectl context
3. Export the KIND kubeconfig:
   ```bash
   kind get kubeconfig --name llm-rag-cluster > ~/.kube/kind-config
   export KUBECONFIG=~/.kube/kind-config
   ```

### Pod Startup Issues

If pods are not starting or are crashing:

1. Check the pod logs
2. Verify the ConfigMap and PVC are correctly created
3. Ensure the Docker image is accessible to the cluster
