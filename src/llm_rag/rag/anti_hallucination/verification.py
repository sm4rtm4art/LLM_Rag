"""Combined verification strategies for anti-hallucination.

This module provides functions that combine multiple verification strategies
to achieve more robust hallucination detection.
"""

import logging
from typing import List, Optional, Tuple

from llm_rag.rag.anti_hallucination.config import HallucinationConfig
from llm_rag.rag.anti_hallucination.entity import verify_entities_in_context
from llm_rag.rag.anti_hallucination.similarity import embedding_based_verification

logger = logging.getLogger(__name__)


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
    """Combine entity and embedding verification for robust fact-checking.

    Args:
        response: Generated response
        context: Context used for generation
        config: HallucinationConfig object with all settings
        entity_threshold: Minimum ratio for entity check (overrides config)
        embedding_threshold: Minimum cosine similarity (overrides config)
        model_name: SentenceTransformer model to use (overrides config)
        threshold: Backward compatibility value for both checks
        languages: List of language codes for stopwords

    Returns:
        Tuple: (is_verified, entity_coverage, embeddings_sim, missing_entities)

    """
    # Use config object if provided, otherwise create default
    if config is None:
        config = HallucinationConfig()

    # Parameter precedence: explicit args > threshold > config defaults
    e_threshold = entity_threshold or threshold or config.entity_threshold
    m_name = model_name or config.model_name

    # Get entity verification results
    is_verified_entity, entity_cov, missing_entities = verify_entities_in_context(
        response, context, threshold=e_threshold, languages=languages
    )

    # Perform embedding verification if enabled
    if config.use_embeddings:
        emb_threshold = embedding_threshold or threshold or config.embedding_threshold
        is_verified_embed, embeddings_sim = embedding_based_verification(
            response, context, threshold=emb_threshold, model_name=m_name
        )
    else:
        is_verified_embed, embeddings_sim = True, 1.0

    # Combine results
    overall_verified = is_verified_entity and is_verified_embed

    # Log detailed verification results
    logger.debug(
        'Verification results: entity_coverage=%.2f, embedding_sim=%.2f, '
        'entities_verified=%s, embeddings_verified=%s, overall=%s',
        entity_cov,
        embeddings_sim,
        is_verified_entity,
        is_verified_embed,
        overall_verified,
    )

    return overall_verified, entity_cov, embeddings_sim, missing_entities
