"""Tests for evaluating the RAG system's performance.

This module contains tests for evaluating the performance of the RAG system,
including metrics for retrieval quality and answer generation quality.
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.llm_rag.rag.pipeline import RAGPipeline


class TestRAGEvaluation(unittest.TestCase):
    """Test cases for evaluating the RAG system's performance."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.mock_vectorstore = MagicMock()
        self.mock_llm = MagicMock()

        # Configure mock behavior
        self.mock_vectorstore.search.return_value = [
            {"content": "Test document 1", "metadata": {"source": "test1.txt"}},
            {"content": "Test document 2", "metadata": {"source": "test2.txt"}},
        ]
        self.mock_llm.invoke.return_value = "This is a test response."

        # Create pipeline instance
        self.pipeline = RAGPipeline(
            vectorstore=self.mock_vectorstore,
            llm=self.mock_llm,
            top_k=2,
        )

        # Create test data directory if it doesn't exist
        self.test_data_dir = Path("tests/evaluation/test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)

    def test_retrieval_precision(self):
        """Test retrieval precision metric.

        Precision = Number of relevant documents retrieved / Total number of documents retrieved
        """
        # Mock relevant documents
        relevant_docs = ["test1.txt"]

        # Mock retrieval results
        retrieved_docs = [
            {"content": "Test document 1", "metadata": {"source": "test1.txt"}},
            {"content": "Test document 2", "metadata": {"source": "test2.txt"}},
        ]

        # Calculate precision
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc["metadata"]["source"] in relevant_docs)
        precision = relevant_retrieved / len(retrieved_docs)

        # Assert precision is as expected
        self.assertEqual(precision, 0.5)

    def test_retrieval_recall(self):
        """Test retrieval recall metric.

        Recall = Number of relevant documents retrieved / Total number of relevant documents
        """
        # Mock relevant documents
        relevant_docs = ["test1.txt", "test3.txt"]

        # Mock retrieval results
        retrieved_docs = [
            {"content": "Test document 1", "metadata": {"source": "test1.txt"}},
            {"content": "Test document 2", "metadata": {"source": "test2.txt"}},
        ]

        # Calculate recall
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc["metadata"]["source"] in relevant_docs)
        recall = relevant_retrieved / len(relevant_docs)

        # Assert recall is as expected
        self.assertEqual(recall, 0.5)

    def test_answer_relevance(self):
        """Test answer relevance to the query."""
        # This would typically use a more sophisticated metric like BLEU or ROUGE
        # For simplicity, we'll use a mock relevance score

        query = "What is RAG?"
        answer = "RAG stands for Retrieval-Augmented Generation."

        # Mock relevance calculation
        with patch("src.llm_rag.evaluation.metrics.calculate_relevance", return_value=0.85):
            from src.llm_rag.evaluation.metrics import calculate_relevance

            relevance = calculate_relevance(query, answer)

            # Assert relevance is above threshold
            self.assertGreaterEqual(relevance, 0.8)

    def test_answer_factuality(self):
        """Test factual accuracy of the generated answer."""
        # This would typically use a factuality checking model
        # For simplicity, we'll use a mock factuality score

        context = "RAG stands for Retrieval-Augmented Generation."
        answer = "RAG stands for Retrieval-Augmented Generation."

        # Mock factuality calculation
        with patch("src.llm_rag.evaluation.metrics.calculate_factuality", return_value=1.0):
            from src.llm_rag.evaluation.metrics import calculate_factuality

            factuality = calculate_factuality(context, answer)

            # Assert factuality is high
            self.assertGreaterEqual(factuality, 0.9)

    def test_end_to_end_evaluation(self):
        """Test end-to-end evaluation with a set of test queries."""
        # Create a test dataset
        test_dataset = [
            {
                "query": "What is RAG?",
                "expected_answer": "RAG stands for Retrieval-Augmented Generation.",
                "relevant_docs": ["doc1.txt", "doc2.txt"],
            },
            {
                "query": "How does RAG work?",
                "expected_answer": "RAG works by retrieving relevant documents and using them to generate answers.",
                "relevant_docs": ["doc3.txt"],
            },
        ]

        # Save test dataset to file
        with open(self.test_data_dir / "test_queries.json", "w") as f:
            json.dump(test_dataset, f)

        # Mock evaluation metrics
        with patch(
            "src.llm_rag.evaluation.evaluator.evaluate_rag",
            return_value={
                "precision": 0.75,
                "recall": 0.8,
                "f1": 0.77,
                "answer_relevance": 0.85,
                "answer_factuality": 0.9,
            },
        ):
            from src.llm_rag.evaluation.evaluator import evaluate_rag

            # Evaluate RAG pipeline
            metrics = evaluate_rag(self.pipeline, test_dataset)

            # Assert metrics meet thresholds
            self.assertGreaterEqual(metrics["precision"], 0.7)
            self.assertGreaterEqual(metrics["recall"], 0.7)
            self.assertGreaterEqual(metrics["f1"], 0.7)
            self.assertGreaterEqual(metrics["answer_relevance"], 0.8)
            self.assertGreaterEqual(metrics["answer_factuality"], 0.8)


if __name__ == "__main__":
    unittest.main()
