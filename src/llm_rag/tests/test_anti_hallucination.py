"""Tests for the anti_hallucination module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from llm_rag.rag.anti_hallucination import (
    HallucinationConfig,
    advanced_verify_response,
    calculate_hallucination_score,
    embedding_based_verification,
    extract_key_entities,
    needs_human_review,
    post_process_response,
    verify_entities_in_context,
)


class TestAntiHallucination(unittest.TestCase):
    """Test suite for anti_hallucination module."""

    def test_extract_key_entities_empty_text(self):
        """Test that empty text returns empty set."""
        self.assertEqual(extract_key_entities(""), set())

    def test_extract_key_entities_only_stopwords(self):
        """Test that text with only stopwords returns empty set."""
        self.assertEqual(extract_key_entities("the an a if"), set())

    def test_extract_key_entities_normal_text(self):
        """Test that normal text returns expected entities."""
        text = "Machine learning models can process information quickly."
        entities = extract_key_entities(text)
        self.assertIn("machine", entities)
        self.assertIn("learning", entities)
        self.assertIn("models", entities)
        self.assertIn("process", entities)
        self.assertIn("information", entities)
        self.assertIn("quickly", entities)

    def test_extract_key_entities_din_standards(self):
        """Test that DIN standards are properly extracted."""
        text = "According to DIN EN ISO 9001, quality management is important."
        entities = extract_key_entities(text)
        self.assertIn("dinen9001", entities) if "dinen9001" in entities else self.assertIn("iso9001", entities)
        self.assertIn("quality", entities)
        self.assertIn("management", entities)
        self.assertIn("important", entities)

    def test_verify_entities_perfect_match(self):
        """Test verification with perfect match."""
        response = "Machine learning is important"
        context = "Machine learning is an important field of AI"
        is_verified, coverage, missing = verify_entities_in_context(response, context, threshold=0.7)
        self.assertTrue(is_verified)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(missing, [])

    def test_verify_entities_partial_match(self):
        """Test verification with partial match."""
        response = "Machine learning and quantum computing are related"
        context = "Machine learning is an important field of AI"
        is_verified, coverage, missing = verify_entities_in_context(response, context, threshold=0.7)
        self.assertFalse(is_verified)
        self.assertIn("quantum", missing)
        self.assertIn("computing", missing)
        self.assertIn("related", missing)
        self.assertLess(coverage, 0.7)

    def test_verify_entities_empty_response(self):
        """Test verification with empty response."""
        response = ""
        context = "Machine learning is an important field of AI"
        is_verified, coverage, missing = verify_entities_in_context(response, context)
        self.assertTrue(is_verified)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(missing, [])

    @patch("llm_rag.rag.anti_hallucination.get_sentence_transformer_model")
    @patch("llm_rag.rag.anti_hallucination.cosine_similarity")
    def test_embedding_verification(self, mock_cosine, mock_get_model):
        """Test embedding-based verification."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5, 0.5]])
        mock_get_model.return_value = mock_model
        mock_cosine.return_value = np.array([[0.8]])

        is_verified, sim = embedding_based_verification("test response", "test context", threshold=0.7)

        self.assertTrue(is_verified)
        self.assertEqual(sim, 0.8)
        mock_model.encode.assert_any_call(["test response"])
        mock_model.encode.assert_any_call(["test context"])

    @patch("llm_rag.rag.anti_hallucination.get_sentence_transformer_model")
    @patch("llm_rag.rag.anti_hallucination.cosine_similarity")
    def test_embedding_verification_below_threshold(self, mock_cosine, mock_get_model):
        """Test embedding verification below threshold."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5, 0.5]])
        mock_get_model.return_value = mock_model
        mock_cosine.return_value = np.array([[0.5]])

        is_verified, sim = embedding_based_verification("test response", "test context", threshold=0.7)

        self.assertFalse(is_verified)
        self.assertEqual(sim, 0.5)

    @patch("llm_rag.rag.anti_hallucination.embedding_based_verification")
    @patch("llm_rag.rag.anti_hallucination.verify_entities_in_context")
    def test_advanced_verify_response(self, mock_verify, mock_embed):
        """Test the advanced verify response function."""
        # Setup mocks
        mock_verify.return_value = (True, 0.9, [])
        mock_embed.return_value = (True, 0.85)

        verified, entity_cov, embed_sim, missing = advanced_verify_response("test response", "test context")

        self.assertTrue(verified)
        self.assertEqual(entity_cov, 0.9)
        self.assertEqual(embed_sim, 0.85)
        self.assertEqual(missing, [])

    def test_calculate_hallucination_score(self):
        """Test hallucination score calculation."""
        # Entity coverage only
        score = calculate_hallucination_score(0.8)
        self.assertEqual(score, 0.8)

        # Entity coverage and embedding similarity
        score = calculate_hallucination_score(0.8, 0.9, entity_weight=0.6)
        self.assertEqual(score, 0.8 * 0.6 + 0.9 * 0.4)

    def test_needs_human_review(self):
        """Test human review flagging."""
        # Score above threshold
        config = HallucinationConfig(
            human_review_threshold=0.5, entity_critical_threshold=0.3, embedding_critical_threshold=0.4
        )

        # Test with score well above threshold and good coverage
        self.assertFalse(needs_human_review(0.7, config, entity_coverage=0.8, embedding_similarity=0.8))

        # Score below threshold
        self.assertTrue(needs_human_review(0.4, config))

        # Entity coverage below critical threshold
        self.assertTrue(needs_human_review(0.7, config, entity_coverage=0.2, entity_critical_threshold=0.3))

        # Embedding similarity below critical threshold
        self.assertTrue(
            needs_human_review(
                0.7, config, entity_coverage=0.8, embedding_similarity=0.3, embedding_critical_threshold=0.4
            )
        )

    @patch("llm_rag.rag.anti_hallucination.advanced_verify_response")
    def test_post_process_response_basic(self, mock_advanced):
        """Test basic post-processing functionality."""
        mock_advanced.return_value = (True, 0.9, 0.85, [])

        # Simple case, no warnings
        result = post_process_response("response", "context")
        self.assertEqual(result, "response")

    @patch("llm_rag.rag.anti_hallucination.advanced_verify_response")
    def test_post_process_response_with_warnings(self, mock_advanced):
        """Test post-processing with warnings."""
        mock_advanced.return_value = (False, 0.5, 0.7, ["hallucinated", "term"])

        # Should add warning message
        result = post_process_response("response", "context")
        self.assertIn("SYSTEM WARNING", result)
        self.assertIn("hallucinated", result)
        self.assertIn("term", result)

    @patch("llm_rag.rag.anti_hallucination.advanced_verify_response")
    def test_post_process_response_with_metadata(self, mock_advanced):
        """Test post-processing with metadata return."""
        mock_advanced.return_value = (True, 0.9, 0.85, [])

        response, metadata = post_process_response("response", "context", return_metadata=True)

        self.assertEqual(response, "response")
        self.assertTrue(metadata["verified"])
        self.assertEqual(metadata["entity_coverage"], 0.9)
        self.assertEqual(metadata["embedding_similarity"], 0.85)
        self.assertEqual(metadata["missing_entities"], [])

    @patch("llm_rag.rag.anti_hallucination.advanced_verify_response")
    @patch("llm_rag.rag.anti_hallucination.needs_human_review")
    def test_post_process_response_with_human_review(self, mock_needs, mock_advanced):
        """Test post-processing with human review flagging."""
        mock_advanced.return_value = (True, 0.7, 0.6, [])
        mock_needs.return_value = True

        # Mock the generate_verification_warning function to include "expert review"
        with patch("llm_rag.rag.anti_hallucination.generate_verification_warning") as mock_warning:
            mock_warning.return_value = "This response has been flagged for expert review"

            config = HallucinationConfig(flag_for_human_review=True)
            response, metadata = post_process_response("response", "context", config=config, return_metadata=True)

            self.assertIn("expert review", response)
            self.assertTrue(metadata["human_review_recommended"])


if __name__ == "__main__":
    unittest.main()
