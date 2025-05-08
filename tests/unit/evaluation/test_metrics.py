"""Unit tests for the metrics module.

This module contains tests for the evaluation metrics functions,
including relevance and factuality calculations.
"""

import sys
from unittest.mock import MagicMock

from llm_rag.evaluation.metrics import calculate_factuality, calculate_relevance

# Nur die problematischen Module mocken, bevor sie importiert werden
if 'sentence_transformers' not in sys.modules:
    sys.modules['sentence_transformers'] = MagicMock()
    sys.modules['sentence_transformers.SentenceTransformer'] = MagicMock()

if 'torch' not in sys.modules:
    sys.modules['torch'] = MagicMock()


class TestCalculateRelevance:
    """Test cases for the calculate_relevance function."""

    def test_relevance_with_matching_terms(self) -> None:
        """Test relevance calculation with matching terms in query and answer."""
        # Arrange
        query = 'What is retrieval augmented generation?'
        answer = 'Retrieval augmented generation is a technique that combines retrieval and generation.'

        # Act
        score = calculate_relevance(query, answer)

        # Assert
        assert score > 0.0
        assert score <= 1.0
        # "retrieval" and "augmented" and "generation" are in both query and answer
        assert score >= 0.5  # At least some overlap should be detected

    def test_relevance_with_no_overlap(self) -> None:
        """Test relevance calculation with no term overlap."""
        # Arrange
        query = 'What is RAG?'
        answer = 'The system combines different techniques for improved results.'

        # Act
        score = calculate_relevance(query, answer)

        # Assert
        assert score == 0.0  # No overlap between terms

    def test_relevance_with_empty_answer(self) -> None:
        """Test relevance calculation with empty answer."""
        # Arrange
        query = 'What is RAG?'
        answer = ''

        # Act
        score = calculate_relevance(query, answer)

        # Assert
        assert score == 0.0  # Empty answer should result in zero score

    def test_relevance_with_empty_query(self) -> None:
        """Test relevance calculation with empty query."""
        # Arrange
        query = ''
        answer = 'RAG is retrieval augmented generation.'

        # Act
        score = calculate_relevance(query, answer)

        # Assert
        assert score == 0.5  # Default score for empty query should be 0.5

    def test_relevance_with_model(self) -> None:
        """Test relevance calculation with a model parameter."""
        # Arrange
        query = 'What is RAG?'
        answer = 'RAG is retrieval augmented generation.'
        mock_model = MagicMock()

        # Act - the current implementation ignores the model
        score = calculate_relevance(query, answer, model=mock_model)

        # Assert
        assert score > 0.0  # Should still calculate a score
        # Die aktuelle Implementierung verwendet den Model-Parameter nicht,
        # aber der Test stellt sicher, dass die Funktion trotzdem funktioniert

    def test_relevance_with_context(self) -> None:
        """Test relevance calculation with context parameter."""
        # Arrange
        query = 'What is RAG?'
        answer = 'RAG is retrieval augmented generation.'
        context = 'RAG stands for Retrieval Augmented Generation which is a technique...'

        # Act - the current implementation ignores context but we should test it
        score = calculate_relevance(query, answer, context=context)

        # Assert
        assert score > 0.0  # Should still calculate a score based on query and answer


class TestCalculateFactuality:
    """Test cases for the calculate_factuality function."""

    def test_factuality_with_reference(self) -> None:
        """Test factuality calculation with reference answer."""
        # Arrange
        answer = 'RAG is retrieval augmented generation.'
        reference = 'RAG stands for Retrieval Augmented Generation.'

        # Act
        score = calculate_factuality(answer, reference=reference)

        # Assert
        assert score > 0.0
        assert score <= 1.0
        # Both answers contain similar terms
        assert score >= 0.5

    def test_factuality_with_context(self) -> None:
        """Test factuality calculation with context."""
        # Arrange
        answer = 'RAG is retrieval augmented generation.'
        context = 'RAG (Retrieval Augmented Generation) is a technique that...'

        # Act
        score = calculate_factuality(answer, context=context)

        # Assert
        assert score > 0.0
        assert score <= 1.0
        # Answer terms appear in context
        assert score >= 0.5

    def test_factuality_with_no_reference_or_context(self) -> None:
        """Test factuality calculation with neither reference nor context."""
        # Arrange
        answer = 'RAG is retrieval augmented generation.'

        # Act
        score = calculate_factuality(answer)

        # Assert
        assert score == 0.5  # Default score when no reference or context is provided

    def test_factuality_with_empty_answer(self) -> None:
        """Test factuality calculation with empty answer."""
        # Arrange
        answer = ''
        reference = 'RAG is retrieval augmented generation.'

        # Act
        score = calculate_factuality(answer, reference=reference)

        # Assert
        assert score == 0.0  # Empty answer should result in zero score

    def test_factuality_with_empty_reference(self) -> None:
        """Test factuality calculation with empty reference."""
        # Arrange
        answer = 'RAG is retrieval augmented generation.'
        reference = ''

        # Act
        score = calculate_factuality(answer, reference=reference)

        # Assert
        assert score == 0.5  # Default score for empty reference

    def test_factuality_with_empty_context(self) -> None:
        """Test factuality calculation with empty context."""
        # Arrange
        answer = 'RAG is retrieval augmented generation.'
        context = ''

        # Act
        score = calculate_factuality(answer, context=context)

        # Assert
        assert score == 0.5  # Default score for empty context

    def test_factuality_with_model(self) -> None:
        """Test factuality calculation with model parameter."""
        # Arrange
        answer = 'RAG is retrieval augmented generation.'
        reference = 'RAG stands for Retrieval Augmented Generation.'
        mock_model = MagicMock()

        # Act - current implementation ignores the model
        score = calculate_factuality(answer, reference=reference, model=mock_model)

        # Assert
        assert score > 0.0  # Should still calculate a score
        # Die aktuelle Implementierung verwendet den Model-Parameter nicht,
        # aber der Test stellt sicher, dass die Funktion trotzdem funktioniert
