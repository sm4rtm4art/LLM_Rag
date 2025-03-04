"""Evaluation module for the LLM RAG system.

This package contains modules for evaluating the performance of the RAG system,
including metrics for answer relevance, factuality, and overall system performance.
"""

from .evaluator import evaluate_rag
from .metrics import calculate_factuality, calculate_relevance

__all__ = ["calculate_relevance", "calculate_factuality", "evaluate_rag"]
