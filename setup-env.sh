#!/bin/bash
export KUBECONFIG=/tmp/kind-kubeconfig
kubectl config get-contexts
kubectl cluster-info --context kind-llm-rag-test
kubectl get nodes
