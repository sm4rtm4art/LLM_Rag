"""Unit tests for the evaluator module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_rag.evaluation.evaluator import evaluate_rag


class TestEvaluateRAG:
    """Test cases for the evaluate_rag function."""

    def test_evaluate_rag_with_dict_data(self) -> None:
        """Test evaluate_rag function with dictionary test data."""
        # Arrange
        test_data = [
            {
                'query': 'What is RAG?',
                'expected_answer': 'RAG is Retrieval Augmented Generation.',
                'relevant_docs': ['doc1', 'doc2'],
            }
        ]

        # Create mock RAG pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = 'RAG is Retrieval Augmented Generation.'
        mock_pipeline.retrieved_documents = [
            MagicMock(metadata={'source': 'doc1'}),
            MagicMock(metadata={'source': 'doc3'}),
        ]
        mock_pipeline.format_context.return_value = 'Context about RAG'

        # Act
        with patch('llm_rag.evaluation.evaluator.calculate_relevance', return_value=0.8) as mock_relevance:
            with patch('llm_rag.evaluation.evaluator.calculate_factuality', return_value=0.9) as mock_factuality:
                results = evaluate_rag(test_data, mock_pipeline)

        # Assert
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 'answer_relevance' in results
        assert 'answer_factuality' in results

        # Precision: 1/2 = 0.5, Recall: 1/2 = 0.5, F1: 2*0.5*0.5/(0.5+0.5) = 0.5
        assert results['precision'] == 0.5
        assert results['recall'] == 0.5
        assert results['f1'] == 0.5
        assert results['answer_relevance'] == 0.8
        assert results['answer_factuality'] == 0.9

        # Verify mocks were called correctly
        mock_pipeline.generate.assert_called_once_with('What is RAG?')
        mock_relevance.assert_called_once()
        mock_factuality.assert_called_once()

    def test_evaluate_rag_with_file_path(self, tmp_path: Path) -> None:
        """Test evaluate_rag function with a file path as input."""
        # Arrange
        test_data = [
            {
                'query': 'What is RAG?',
                'expected_answer': 'RAG is Retrieval Augmented Generation.',
                'relevant_docs': ['doc1', 'doc2'],
            }
        ]

        # Create a temporary JSON file
        test_file = tmp_path / 'test_data.json'
        with open(test_file, 'w') as f:
            json.dump(test_data, f)

        # Create mock RAG pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = 'RAG is a system.'

        # Korrigiert für Precision=0.5: 2 Dokumente zurückgegeben, aber nur 1 ist relevant
        mock_pipeline.retrieved_documents = [
            MagicMock(metadata={'source': 'doc1'}),
            MagicMock(metadata={'source': 'doc3'}),  # Dieses Dokument ist nicht in relevant_docs
        ]

        # Mock the metrics calculation
        with patch('llm_rag.evaluation.evaluator.calculate_relevance', return_value=0.7) as mock_relevance:
            with patch('llm_rag.evaluation.evaluator.calculate_factuality', return_value=0.6) as mock_factuality:
                results = evaluate_rag(test_file, mock_pipeline)

        # Assert
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert results['precision'] == 0.5  # 1/2 = 0.5
        assert results['recall'] == 0.5  # 1/2 = 0.5
        assert results['f1'] == 0.5  # 2*0.5*0.5/(0.5+0.5) = 0.5

        # Verify mocks were called correctly
        mock_pipeline.generate.assert_called_once_with('What is RAG?')
        mock_relevance.assert_called_once()
        mock_factuality.assert_called_once()
