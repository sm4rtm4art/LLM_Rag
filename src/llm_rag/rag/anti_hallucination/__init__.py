"""Anti-hallucination module for detecting and mitigating hallucinations in LLM responses.

This package provides functions and classes for:
1. Entity-based verification
2. Embedding-based similarity checks
3. Combined hallucination scoring
4. Post-processing of responses with warnings
"""

# Re-export all public components
from llm_rag.rag.anti_hallucination.config import HallucinationConfig
from llm_rag.rag.anti_hallucination.entity import extract_key_entities, verify_entities_in_context
from llm_rag.rag.anti_hallucination.processing import (
    calculate_hallucination_score,
    generate_verification_warning,
    needs_human_review,
    post_process_response,
)
from llm_rag.rag.anti_hallucination.similarity import embedding_based_verification, get_sentence_transformer_model
from llm_rag.rag.anti_hallucination.utils import load_stopwords

# Advanced combined verification
from llm_rag.rag.anti_hallucination.verification import advanced_verify_response

__all__ = [
    # Config
    "HallucinationConfig",
    # Entity verification
    "extract_key_entities",
    "verify_entities_in_context",
    # Similarity verification
    "embedding_based_verification",
    "get_sentence_transformer_model",
    # Processing and scoring
    "calculate_hallucination_score",
    "generate_verification_warning",
    "needs_human_review",
    "post_process_response",
    # Combined verification
    "advanced_verify_response",
    # Utilities
    "load_stopwords",
]
