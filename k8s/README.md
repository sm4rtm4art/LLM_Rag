# Kubernetes Deployment for LLM-RAG

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

The CI/CD pipeline uses these configuration files to deploy the application to test, development, and staging environments. The pipeline will:

1. Build the Docker image
2. Push it to Docker Hub
3. Deploy it to the appropriate environment

For local development and testing, use the scripts provided above.

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
