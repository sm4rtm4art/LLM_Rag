# Kubernetes Testing Utilities

This directory contains utilities to support lightweight Kubernetes testing in GitHub Actions.

## Overview

The main issue being solved is the GitHub Actions runner running out of disk space during Kubernetes tests when using the full application Docker image:

```
Error response from daemon: write /var/lib/docker/tmp/docker-export-2128228893/blobs/sha256/5ff5997c8e9fa88af8321fc3ecbb4ea00fba624ebeff51c3118c1de0fe8fbfc6: no space left on device
```

## Solution Components

1. **Lightweight Test Dockerfile (`Dockerfile.k8s-test`)**:

   - A minimal Docker image containing only what's needed to test Kubernetes connectivity
   - Uses a mock API implementation instead of the full LLM-RAG application
   - Significantly reduces image size for Kubernetes testing purposes

2. **Mock API Implementation (`mock_api.py`)**:

   - Provides the same API interface as the real application
   - Returns mock responses for `/query` endpoints
   - Implements `/health` endpoint for Kubernetes probes

3. **CI/CD Integration**:
   - Modified GitHub workflow to use the lightweight image for Kubernetes tests
   - Added environmental flag `USE_TEST_IMAGE` to control which Dockerfile is used

## How to Use

For local testing:

```bash
# Build the test image
docker build -t llm-rag:test -f Dockerfile.k8s-test .

# Run the test container
docker run -p 8000:8000 llm-rag:test
```

For GitHub Actions:

- The test image will be used automatically during Kubernetes testing

## Notes

- The test image is for validating Kubernetes configuration only
- It does not test actual application functionality, only API connectivity
- For full application testing, use the regular Dockerfile
