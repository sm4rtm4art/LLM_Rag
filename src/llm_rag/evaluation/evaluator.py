"""Evaluator module for assessing RAG system performance.

This module provides functions for evaluating the overall performance of a RAG system,
including metrics like precision, recall, F1 score, answer relevance, and factuality.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .metrics import calculate_factuality, calculate_relevance

logger = logging.getLogger(__name__)


def evaluate_rag(
    test_data: Union[str, Path, List[Dict[str, Any]]], rag_pipeline: Any, metrics: Optional[List[str]] = None, **kwargs
) -> Dict[str, float]:
    """Evaluate a RAG system using a test dataset.

    Args:
        test_data: Path to a JSON file containing test data or a list of test cases
        rag_pipeline: The RAG pipeline to evaluate
        metrics: List of metrics to calculate (default: all available metrics)
        **kwargs: Additional arguments for specific metrics

    Returns:
        Dict[str, float]: A dictionary of evaluation metrics

    """
    logger.info('Starting RAG system evaluation...')

    # Load test data if a path is provided
    if isinstance(test_data, (str, Path)):
        with open(test_data, 'r') as f:
            test_cases = json.load(f)
    else:
        test_cases = test_data

    if not test_cases:
        logger.warning('No test cases provided for evaluation')
        return {}

    # Default metrics if none specified
    if metrics is None:
        metrics = ['precision', 'recall', 'f1', 'answer_relevance', 'answer_factuality']

    # Initialize results
    results = {metric: 0.0 for metric in metrics}
    retrieval_metrics = {'precision': [], 'recall': [], 'f1': []}
    generation_metrics = {'answer_relevance': [], 'answer_factuality': []}

    # Process each test case
    for i, test_case in enumerate(test_cases):
        query = test_case.get('query', '')
        expected_answer = test_case.get('expected_answer', '')
        relevant_docs = test_case.get('relevant_docs', [])

        logger.info(f'Evaluating test case {i + 1}/{len(test_cases)}: {query[:50]}...')

        # Run the RAG pipeline
        try:
            # Set test mode if available
            if hasattr(rag_pipeline, '_test_mode'):
                rag_pipeline._test_mode = True

            # Get the answer and retrieved documents
            answer = rag_pipeline.generate(query)
            retrieved_docs = getattr(rag_pipeline, 'retrieved_documents', [])

            # Calculate retrieval metrics if relevant_docs are provided
            if relevant_docs and 'precision' in metrics:
                retrieved_ids = [doc.metadata.get('source', '') for doc in retrieved_docs]
                tp = len(set(retrieved_ids).intersection(set(relevant_docs)))
                precision = tp / len(retrieved_ids) if retrieved_ids else 0
                recall = tp / len(relevant_docs) if relevant_docs else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                retrieval_metrics['precision'].append(precision)
                retrieval_metrics['recall'].append(recall)
                retrieval_metrics['f1'].append(f1)

            # Calculate generation metrics
            if 'answer_relevance' in metrics:
                relevance = calculate_relevance(query, answer, **kwargs)
                generation_metrics['answer_relevance'].append(relevance)

            if 'answer_factuality' in metrics:
                context = rag_pipeline.format_context(retrieved_docs) if hasattr(rag_pipeline, 'format_context') else ''
                factuality = calculate_factuality(answer, context, expected_answer, **kwargs)
                generation_metrics['answer_factuality'].append(factuality)

        except Exception as e:
            logger.error(f'Error evaluating test case {i + 1}: {str(e)}')

    # Calculate average metrics
    for metric in retrieval_metrics:
        if metric in metrics and retrieval_metrics[metric]:
            results[metric] = sum(retrieval_metrics[metric]) / len(retrieval_metrics[metric])

    for metric in generation_metrics:
        if metric in metrics and generation_metrics[metric]:
            results[metric] = sum(generation_metrics[metric]) / len(generation_metrics[metric])

    logger.info(f'Evaluation complete. Results: {results}')
    return results
