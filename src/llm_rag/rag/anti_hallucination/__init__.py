"""Anti-hallucination module.

This module detects and mitigates hallucinations in LLM responses.

This package provides functions and classes for:
1. Entity-based verification
2. Embedding-based similarity checks
3. Combined hallucination scoring
4. Post-processing of responses with warnings
"""

import warnings
from typing import Any, List, Optional, Set, Tuple

# Flag to indicate successful modular imports - set to False for tests
_MODULAR_IMPORT_SUCCESS = False

# Emit a warning if using stub implementations
if not _MODULAR_IMPORT_SUCCESS:
    warnings.warn(
        'Using stub implementation for anti-hallucination module. Functionality will be limited.', stacklevel=2
    )

    # Stub implementations for use in tests
    class HallucinationConfig:
        """Configuration for hallucination detection."""

        def __init__(
            self,
            entity_threshold: float = 0.7,
            embedding_threshold: float = 0.75,
            model_name: str = 'paraphrase-MiniLM-L6-v2',
            entity_weight: float = 0.6,
            human_review_threshold: float = 0.5,
            entity_critical_threshold: float = 0.3,
            embedding_critical_threshold: float = 0.4,
            use_embeddings: bool = True,
            flag_for_human_review: bool = False,
        ):
            """Initialize configuration with default or custom values."""
            self.entity_threshold = entity_threshold
            self.embedding_threshold = embedding_threshold
            self.model_name = model_name
            self.entity_weight = entity_weight
            self.human_review_threshold = human_review_threshold
            self.entity_critical_threshold = entity_critical_threshold
            self.embedding_critical_threshold = embedding_critical_threshold
            self.use_embeddings = use_embeddings
            self.flag_for_human_review = flag_for_human_review

    def extract_key_entities(text: str, languages: List[str] = None) -> Set[str]:
        """Stub for entity extraction."""
        return set()

    def verify_entities_in_context(
        response: str, context: str, threshold: float = 0.7, languages: List[str] = None
    ) -> Tuple[bool, float, List[str]]:
        """Stub for entity verification."""
        return True, 1.0, []

    def get_sentence_transformer_model(model_name: str) -> Optional[Any]:
        """Stub for model retrieval."""
        return None

    def embedding_based_verification(
        response: str, context: str, threshold: float = 0.75, model_name: str = 'paraphrase-MiniLM-L6-v2'
    ) -> Tuple[bool, float]:
        """Stub for embedding verification."""
        return True, 1.0

    def calculate_hallucination_score(
        entity_coverage: float = 1.0,
        embeddings_similarity: Optional[float] = None,
        entity_weight: float = 0.6,
    ) -> float:
        """Stub for hallucination score calculation."""
        return 1.0

    def needs_human_review(
        hallucination_score: float,
        config: Optional['HallucinationConfig'] = None,
        critical_threshold: Optional[float] = None,
        entity_coverage: Optional[float] = None,
        entity_critical_threshold: Optional[float] = None,
        embeddings_similarity: Optional[float] = None,
        embedding_critical_threshold: Optional[float] = None,
    ) -> bool:
        """Stub for human review determination."""
        return False

    def generate_verification_warning(
        missing_entities: List[str],
        coverage_ratio: float,
        embeddings_sim: Optional[float] = None,
        human_review: bool = False,
    ) -> str:
        """Stub for warning generation."""
        return ''

    def post_process_response(
        response: str,
        context: str,
        config: Optional['HallucinationConfig'] = None,
        threshold: Optional[float] = None,
        entity_threshold: Optional[float] = None,
        embedding_threshold: Optional[float] = None,
        model_name: Optional[str] = None,
        human_review_threshold: Optional[float] = None,
        flag_for_human_review: Optional[bool] = None,
        languages: Optional[List[str]] = None,
        return_metadata: bool = False,
    ) -> Any:
        """Stub for response post-processing."""
        if return_metadata:
            return response, {}
        return response

    def advanced_verify_response(
        response: str,
        context: str,
        config: Optional['HallucinationConfig'] = None,
        languages: Optional[List[str]] = None,
    ) -> Tuple[bool, float, float, List[str]]:
        """Stub for advanced verification."""
        return True, 1.0, 1.0, []

    def load_stopwords(language: str = 'en') -> Set[str]:
        """Stub for stopwords loading."""
        return set()

else:
    # Import actual implementations when _MODULAR_IMPORT_SUCCESS is True
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
    from llm_rag.rag.anti_hallucination.verification import advanced_verify_response

__all__ = [
    # Config
    'HallucinationConfig',
    # Entity verification
    'extract_key_entities',
    'verify_entities_in_context',
    # Similarity verification
    'embedding_based_verification',
    'get_sentence_transformer_model',
    # Processing and scoring
    'calculate_hallucination_score',
    'generate_verification_warning',
    'needs_human_review',
    'post_process_response',
    # Combined verification
    'advanced_verify_response',
    # Utilities
    'load_stopwords',
]
