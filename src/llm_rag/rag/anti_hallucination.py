#!/usr/bin/env python
# flake8: noqa: E501
"""Anti-hallucination techniques for RAG systems.

This module provides functions and classes to detect and mitigate hallucinations
in LLM responses. It includes methods for entity verification, embedding-based
similarity checks, and response confidence scoring.

Note: This file is maintained for backward compatibility. For new development,
please use the modular components in the anti_hallucination/ directory.
"""

import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    # First try to import from the modular implementation
    from llm_rag.rag.anti_hallucination import (
        HallucinationConfig,
        advanced_verify_response,
        calculate_hallucination_score,
        embedding_based_verification,
        extract_key_entities,
        generate_verification_warning,
        get_sentence_transformer_model,
        load_stopwords,
        needs_human_review,
        post_process_response,
        verify_entities_in_context,
    )

    # Flag for successful import
    _MODULAR_IMPORT_SUCCESS = True
except ImportError as e:
    # If import fails, use stub/placeholder implementation
    warnings.warn(
        f"Failed to import modular anti-hallucination components: {e}. Using stub implementations.", stacklevel=2
    )
    _MODULAR_IMPORT_SUCCESS = False

    # Placeholder dataclass
    from dataclasses import dataclass

    @dataclass
    class HallucinationConfig:
        """Stub config class."""

        entity_threshold: float = 0.7
        embedding_threshold: float = 0.75
        model_name: str = "paraphrase-MiniLM-L6-v2"
        entity_weight: float = 0.6
        human_review_threshold: float = 0.5
        entity_critical_threshold: float = 0.3
        embedding_critical_threshold: float = 0.4
        use_embeddings: bool = True
        flag_for_human_review: bool = False

    # Stub functions
    def extract_key_entities(text: str, languages: List[str] = None) -> Set[str]:
        """Stub function."""
        return set()

    def verify_entities_in_context(
        response: str, context: str, threshold: float = 0.7, languages: List[str] = None
    ) -> Tuple[bool, float, List[str]]:
        """Stub function."""
        return True, 1.0, []

    def get_sentence_transformer_model(model_name: str) -> Optional[Any]:
        """Stub function."""
        return None

    def embedding_based_verification(
        response: str, context: str, threshold: float = 0.75, model_name: str = "paraphrase-MiniLM-L6-v2"
    ) -> Tuple[bool, float]:
        """Stub function."""
        return True, 1.0

    def advanced_verify_response(
        response: str,
        context: str,
        config: Optional[HallucinationConfig] = None,
        entity_threshold: Optional[float] = None,
        embedding_threshold: Optional[float] = None,
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
        languages: List[str] = None,
    ) -> Tuple[bool, float, float, List[str]]:
        """Stub function."""
        return True, 1.0, 1.0, []

    def calculate_hallucination_score(
        entity_coverage: float, embeddings_similarity: Optional[float] = None, entity_weight: float = 0.6
    ) -> float:
        """Stub function."""
        return 1.0

    def needs_human_review(
        hallucination_score: float,
        config: Optional[HallucinationConfig] = None,
        critical_threshold: Optional[float] = None,
        entity_coverage: float = 0.0,
        entity_critical_threshold: Optional[float] = None,
        embeddings_similarity: Optional[float] = None,
        embedding_critical_threshold: Optional[float] = None,
    ) -> bool:
        """Stub function."""
        return False

    def generate_verification_warning(
        missing_entities: List[str],
        coverage_ratio: float,
        embeddings_sim: Optional[float] = None,
        human_review: bool = False,
    ) -> str:
        """Stub function."""
        return ""

    def post_process_response(
        response: str,
        context: str,
        config: Optional[HallucinationConfig] = None,
        threshold: Optional[float] = None,
        entity_threshold: Optional[float] = None,
        embedding_threshold: Optional[float] = None,
        model_name: Optional[str] = None,
        human_review_threshold: Optional[float] = None,
        flag_for_human_review: Optional[bool] = None,
        return_metadata: bool = False,
        languages: List[str] = None,
    ) -> Any:
        """Stub function."""
        return response if not return_metadata else (response, {})

    def load_stopwords(language: str = "en") -> Set[str]:
        """Stub function."""
        return set()


# Also re-export model cache for backward compatibility
_MODEL_CACHE: Dict[str, Any] = {}

# And _DEFAULT_STOPWORDS
_DEFAULT_STOPWORDS = {
    "en": {"a", "an", "the", "and", "or", "but", "if", "then", "else", "when"},
    "de": {"der", "die", "das", "und", "oder", "aber", "wenn", "dann", "sonst", "wann"},
}

# Suppress deprecation warnings about imports
warnings.filterwarnings("ignore", category=DeprecationWarning, module=__name__)
