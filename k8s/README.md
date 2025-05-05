# Kubernetes Deployment for LLM RAG Application

This directory contains Kubernetes manifests for deploying the LLM RAG application in various environments. The setup provides a robust, scalable, and secure deployment using Kubernetes best practices.

## Components

- **Deployment**: Manages the core application with configurable resources, probes, and security settings
- **Service**: Exposes the application within the cluster
- **Ingress**: Routes external traffic to the application
- **ConfigMap**: Provides configuration for the application
- **PVC**: Provides persistent storage for models and data
- **HPA**: Horizontal Pod Autoscaler for automatic scaling
- **NetworkPolicy**: Secures pod communication

## Environment Configuration

The manifests support multiple environments (dev, staging, production) through parameterization. Environment-specific values can be provided in several ways:

### 1. Using Kustomize

Create a kustomization.yaml file for each environment:

```yaml
# ./overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namePrefix: prod-
commonLabels:
  environment: production
patchesStrategicMerge:
  - deployment-patch.yaml
configMapGenerator:
  - name: llm-rag-env
    behavior: merge
    literals:
      - APP_ENV=production
      - DEBUG_MODE=false
      - MIN_REPLICAS=2
      - MAX_REPLICAS=10
```

### 2. Using Helm

Convert to Helm by replacing variables with template expressions:

```yaml
# Example in deployment.yaml
replicas: {{ .Values.replicaCount }}
image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
```

### 3. Using CI Variables

In CI/CD pipelines, use variable substitution:

```bash
# GitHub Actions example
envsubst < k8s/deployment.yaml > deployment-rendered.yaml
```

## Quick Start

### Local Development with KIND

1. Create a local cluster:

```bash
kind create cluster --config k8s/kind-config.yaml
```

2. Apply manifests:

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### Production Deployment

For production deployments:

1. Adjust resource requests/limits:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "1000m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

2. Enable TLS:

```bash
# Using cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.11.0/cert-manager.yaml
kubectl apply -f k8s/certificate.yaml
```

## Best Practices

### Security

- Non-root user with read-only filesystem
- NetworkPolicy to restrict traffic
- Resource limits to prevent DoS
- Proper liveness/readiness probes
- Seccomp profiles and capability restrictions

### Scaling

- HPA for automatic scaling
- PodDisruptionBudget for reliability
- Anti-affinity rules for distribution

### Monitoring

- Configure logging to stdout/stderr
- Add Prometheus annotations
- Pod lifecycle hooks for graceful shutdown

## Migration to Production

To migrate from CI testing to a production environment:

1. Use a GitOps approach with ArgoCD or Flux
2. Separate CI validation from CD deployment
3. Create environment-specific overlays
4. Replace `${VARIABLE}` placeholders with actual values using kustomize or a templating engine
5. Consider using Helm for more complex deployments

## Troubleshooting

- **Pod crashes**: Check liveness probe, resource limits
- **Ingress issues**: Verify ingress controller, annotations
- **Scaling problems**: Review HPA metrics, resource settings
- **Network connectivity**: Check NetworkPolicy configuration

## Production Checklist

- [ ] Set proper resource requests/limits
- [ ] Enable TLS for Ingress
- [ ] Configure proper node affinity
- [ ] Implement PodDisruptionBudget
- [ ] Set up monitoring and alerts
- [ ] Configure backups for persistent volumes
- [ ] Review security contexts and NetworkPolicies
