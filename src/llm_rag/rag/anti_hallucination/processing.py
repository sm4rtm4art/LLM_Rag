"""Response post-processing for anti-hallucination module.

This module provides functions for scoring, warning generation, and post-processing
of responses to handle potential hallucinations.
"""

import logging
from typing import Any, List, Optional

from llm_rag.rag.anti_hallucination.config import HallucinationConfig
from llm_rag.rag.anti_hallucination.verification import advanced_verify_response

logger = logging.getLogger(__name__)


def calculate_hallucination_score(
    entity_coverage: float, embeddings_similarity: Optional[float] = None, entity_weight: float = 0.6
) -> float:
    """Calculate a combined hallucination score (0-1 scale).

    Lower scores indicate higher likelihood of hallucination.

    Args:
        entity_coverage: Ratio of entities found in context
        embeddings_similarity: Embedding similarity score (if available)
        entity_weight: Weight given to entity coverage vs embedding

    Returns:
        Combined score between 0-1, where lower values indicate possible
        hallucinations

    """
    if embeddings_similarity is None:
        return entity_coverage

    # Combine scores with configurable weighting
    return entity_coverage * entity_weight + embeddings_similarity * (1 - entity_weight)


def needs_human_review(
    hallucination_score: float,
    config: Optional[HallucinationConfig] = None,
    critical_threshold: Optional[float] = None,
    entity_coverage: float = 0.0,
    entity_critical_threshold: Optional[float] = None,
    embeddings_similarity: Optional[float] = None,
    embedding_critical_threshold: Optional[float] = None,
) -> bool:
    """Determine if a response requires human review.

    Args:
        hallucination_score: Combined hallucination score
        config: HallucinationConfig object with thresholds
        critical_threshold: Override for the combined score threshold
        entity_coverage: Entity coverage ratio
        entity_critical_threshold: Override for entity critical threshold
        embeddings_similarity: Embedding similarity score
        embedding_critical_threshold: Override for embedding threshold

    Returns:
        True if human review is recommended

    """
    # Use config or default if not provided
    if config is None:
        config = HallucinationConfig()

    # Use explicit parameters if provided, otherwise use config
    h_threshold = critical_threshold or config.human_review_threshold
    e_threshold = entity_critical_threshold or config.entity_critical_threshold
    emb_threshold = embedding_critical_threshold or config.embedding_critical_threshold

    # Flag as critical if combined score is too low
    if hallucination_score < h_threshold:
        return True

    # Also flag if either individual score is extremely low
    if entity_coverage < e_threshold:
        return True

    if embeddings_similarity is not None and embeddings_similarity < emb_threshold:
        return True

    return False


def generate_verification_warning(
    missing_entities: List[str],
    coverage_ratio: float,
    embeddings_sim: Optional[float] = None,
    human_review: bool = False,
) -> str:
    """Generate a warning message for potentially hallucinated content.

    Args:
        missing_entities: List of entities not found in the context
        coverage_ratio: The ratio of entities that were found in the context
        embeddings_sim: Optional embedding similarity score
        human_review: Whether this response has been flagged for human review

    Returns:
        A warning message

    """
    if not missing_entities:
        return ''

    confidence_level = 'low' if coverage_ratio < 0.5 else 'moderate'

    warning = (
        '\n\n[SYSTEM WARNING: This response may contain hallucinated '
        f'information. Confidence level: {confidence_level}. '
        f'The following terms were not found in the retrieved documents: '
        f'{", ".join(missing_entities[:5])}'
    )

    if len(missing_entities) > 5:
        warning += f' and {len(missing_entities) - 5} more'

    if embeddings_sim is not None:
        warning += f'. Semantic similarity score: {embeddings_sim:.2f}'

    if human_review:
        warning += '. This response has been flagged for expert review'

    warning += '. Please verify this information from other sources.]'

    return warning


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
    """Post-process a response to add warnings about potential hallucinations.

    This function performs verification using entity-based and optionally
    embedding-based methods. It can flag responses for human review and
    return detailed metadata about the verification process.

    Args:
        response: The generated response
        context: The context used to generate the response
        config: HallucinationConfig object with all settings
        threshold: Legacy threshold parameter (for backward compatibility)
        entity_threshold: Override for entity verification threshold
        embedding_threshold: Override for embedding verification threshold
        model_name: Override for SentenceTransformer model name
        human_review_threshold: Override for human review threshold
        flag_for_human_review: Override for human review flagging
        return_metadata: Whether to return metadata along with the response
        languages: List of language codes for stopwords

    Returns:
        Either the processed response string (if return_metadata=False)
        or a tuple of (processed_response, metadata_dict)

    """
    # Use config or create default
    if config is None:
        config = HallucinationConfig()

    # For backward compatibility - simple entity check mode
    if threshold is not None and not return_metadata:
        # No new features, just threshold adjustment
        if (
            entity_threshold is None
            and embedding_threshold is None
            and model_name is None
            and not config.flag_for_human_review
            and not flag_for_human_review
        ):
            from llm_rag.rag.anti_hallucination.entity import verify_entities_in_context

            is_verified, coverage_ratio, missing_entities = verify_entities_in_context(
                response, context, threshold, languages
            )
            if not is_verified:
                warning = generate_verification_warning(missing_entities, coverage_ratio)
                response += warning
            return response

    # Explicit parameters override config settings
    if flag_for_human_review is not None:
        config.flag_for_human_review = flag_for_human_review

    if human_review_threshold is not None:
        config.human_review_threshold = human_review_threshold

    # Advanced verification
    verified, entity_cov, embeddings_sim, missing_entities = advanced_verify_response(
        response=response,
        context=context,
        config=config,
        entity_threshold=entity_threshold,
        embedding_threshold=embedding_threshold,
        model_name=model_name,
        threshold=threshold,
        languages=languages,
    )

    # Calculate hallucination score
    hallucination_score = calculate_hallucination_score(entity_cov, embeddings_sim, config.entity_weight)

    # Check if human review is needed
    requires_review = False
    if config.flag_for_human_review:
        requires_review = needs_human_review(
            hallucination_score=hallucination_score,
            config=config,
            entity_coverage=entity_cov,
            embeddings_similarity=embeddings_sim,
        )

    # Add warning if not verified or requires review
    if not verified or requires_review:
        warning = generate_verification_warning(missing_entities, entity_cov, embeddings_sim, requires_review)
        response += warning

    # Create metadata for caller
    metadata = {
        'verified': verified,
        'entity_coverage': entity_cov,
        'embeddings_similarity': embeddings_sim,
        'missing_entities': missing_entities,
        'hallucination_score': hallucination_score,
        'human_review_recommended': requires_review,
    }

    if return_metadata:
        return response, metadata
    else:
        return response
