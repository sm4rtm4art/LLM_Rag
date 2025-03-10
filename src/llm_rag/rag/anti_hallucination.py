"""Anti-hallucination utilities for RAG systems.

This module reduces hallucinations by combining entity-based and
embedding-based verification. It includes backward compatibility,
model selection flexibility, and caching for performance.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None
    cosine_similarity = None
    np = None

logger = logging.getLogger(__name__)


@dataclass
class HallucinationConfig:
    """Configuration for hallucination detection and verification.

    This class encapsulates all configuration parameters for entity-based
    and embedding-based verification, as well as human review thresholds.
    """

    # Entity verification settings
    entity_threshold: float = 0.7

    # Embedding verification settings
    embedding_threshold: float = 0.75
    model_name: str = "paraphrase-MiniLM-L6-v2"

    # Combined scoring settings
    entity_weight: float = 0.6

    # Human review settings
    human_review_threshold: float = 0.5
    entity_critical_threshold: float = 0.3
    embedding_critical_threshold: float = 0.4

    # Feature flags
    use_embeddings: bool = True
    flag_for_human_review: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if not 0 <= self.entity_threshold <= 1:
            raise ValueError("entity_threshold must be between 0 and 1")
        if not 0 <= self.embedding_threshold <= 1:
            raise ValueError("embedding_threshold must be between 0 and 1")
        if not 0 <= self.entity_weight <= 1:
            raise ValueError("entity_weight must be between 0 and 1")


# Global cache for SentenceTransformer models
_MODEL_CACHE: Dict[str, Any] = {}

# Default stopwords for supported languages
_DEFAULT_STOPWORDS = {
    "en": {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "from",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
    },
    "de": {
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einer",
        "eines",
        "einem",
        "einen",
        "ist",
        "sind",
        "war",
        "waren",
        "wird",
        "werden",
        "wurde",
        "wurden",
        "hat",
        "haben",
        "hatte",
        "hatten",
        "kann",
        "können",
        "konnte",
        "konnten",
        "muss",
        "müssen",
        "musste",
        "mussten",
        "soll",
        "sollen",
        "sollte",
        "sollten",
        "wollen",
        "wollte",
        "wollten",
        "darf",
        "dürfen",
        "durfte",
        "durften",
        "mag",
        "mögen",
        "mochte",
        "mochten",
        "und",
        "oder",
        "aber",
        "denn",
        "weil",
        "wenn",
        "als",
        "ob",
        "damit",
        "obwohl",
        "während",
        "nachdem",
        "bevor",
        "sobald",
        "seit",
        "bis",
        "indem",
        "ohne",
        "außer",
        "gegen",
        "für",
        "mit",
        "zu",
        "von",
        "bei",
        "nach",
        "aus",
        "über",
        "unter",
        "neben",
        "zwischen",
        "vor",
        "hinter",
        "auf",
        "um",
        "herum",
        "durch",
        "entlang",
        "ich",
        "du",
        "er",
        "sie",
        "es",
        "wir",
        "ihr",
        "mich",
        "dich",
        "ihn",
        "uns",
        "euch",
        "ihnen",
        "mein",
        "dein",
        "sein",
        "unser",
        "euer",
        "ihre",
        "ihres",
        "ihrem",
        "ihren",
    },
}


def load_stopwords(language: str = "en") -> Set[str]:
    """Load stopwords for a specific language from configuration files.

    Args:
        language: Language code (e.g., 'en' for English, 'de' for German)

    Returns:
        Set of stopwords for the specified language

    """
    stopwords_path = os.path.join(os.path.dirname(__file__), "resources", f"stopwords_{language}.json")

    try:
        with open(stopwords_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        logger.debug("Could not load stopwords for %s from %s, using defaults", language, stopwords_path)
        return _DEFAULT_STOPWORDS.get(language, set())


def get_sentence_transformer_model(model_name: str) -> Optional[Any]:
    """Retrieve or load a SentenceTransformer model."""
    if SentenceTransformer is None:
        logger.warning("SentenceTransformer is unavailable.")
        return None
    if model_name not in _MODEL_CACHE:
        logger.info("Loading model: %s", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def extract_key_entities(text: str, languages: List[str] = None) -> Set[str]:
    """Extract key entities from text.

    Args:
        text: The text to extract entities from
        languages: List of language codes to load stopwords for (default: ['en'])

    Returns:
        A set of key entities

    """
    if languages is None:
        languages = ["en"]

    # Combine stopwords from all specified languages
    stop_words = set()
    for lang in languages:
        stop_words.update(load_stopwords(lang))

    # Extract words (min 4 chars; includes German umlauts and ß)
    words = re.findall(r"\b[a-zA-ZäöüßÄÖÜ0-9_-]{4,}\b", text.lower())
    entities = {word for word in words if word not in stop_words}

    # Add special handling for DIN standard references
    din_pattern = (
        r"\b(?:DIN|EN|ISO|IEC|CEN|TR)\s*[-_]?\s*\d+"
        r"(?:[-_]\d+)*\b"
    )
    din_refs = re.findall(din_pattern, text)
    entities.update([ref.replace(" ", "").lower() for ref in din_refs])

    return entities


def verify_entities_in_context(
    response: str, context: str, threshold: float = 0.7, languages: List[str] = None
) -> Tuple[bool, float, List[str]]:
    """Verify that entities in the response exist in the context.

    Args:
        response: The generated response
        context: The context used to generate the response
        threshold: The minimum ratio of entities that must be in the context
        languages: List of language codes to use for stopwords

    Returns:
        A tuple of (is_verified, entity_coverage_ratio, missing_entities)

    """
    # Extract entities from response and context
    response_entities = extract_key_entities(response, languages)
    context_entities = extract_key_entities(context, languages)

    if not response_entities:
        return True, 1.0, []

    # Find entities in response that are not in context
    missing_entities = [entity for entity in response_entities if entity not in context_entities]

    # Calculate coverage ratio
    coverage_ratio = 1.0 - (len(missing_entities) / len(response_entities))

    # Determine if the response is verified
    is_verified = coverage_ratio >= threshold

    return is_verified, coverage_ratio, missing_entities


def embedding_based_verification(
    response: str, context: str, threshold: float = 0.75, model_name: str = "paraphrase-MiniLM-L6-v2"
) -> Tuple[bool, float]:
    """Verify response against context using embedding similarity.

    Args:
        response: Generated response.
        context: Context used for generation.
        threshold: Minimum cosine similarity required.
        model_name: Name of the SentenceTransformer model.

    Returns:
        Tuple: (is_verified, similarity_score).

    """
    if SentenceTransformer is None or cosine_similarity is None or np is None:
        logger.warning("Embedding libraries unavailable; skipping check.")
        return True, 1.0

    model = get_sentence_transformer_model(model_name)
    if model is None:
        return True, 1.0

    try:
        response_emb = model.encode([response])
        context_emb = model.encode([context])
        sim = cosine_similarity(response_emb, context_emb)[0][0]
        is_verified = sim >= threshold

        logger.debug("Embedding similarity score: %.4f (threshold: %.2f, verified: %s)", sim, threshold, is_verified)

        return is_verified, sim
    except Exception as e:
        logger.error("Error during embedding verification: %s", str(e))
        # Fail open - assume verification passed when check fails
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
    """Combine entity and embedding verification for robust fact-checking.

    Args:
        response: Generated response.
        context: Context used for generation.
        config: HallucinationConfig object with all settings.
        entity_threshold: Minimum ratio for entity check (overrides config).
        embedding_threshold: Minimum cosine similarity (overrides config).
        model_name: SentenceTransformer model to use (overrides config).
        threshold: Backward compatibility value for both checks.
        languages: List of language codes for stopwords.

    Returns:
        Tuple: (is_verified, entity_coverage, embed_sim, missing_entities).

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
        is_verified_embed, embed_sim = embedding_based_verification(
            response, context, threshold=emb_threshold, model_name=m_name
        )
    else:
        is_verified_embed, embed_sim = True, 1.0

    # Combine results
    overall_verified = is_verified_entity and is_verified_embed

    # Log detailed verification results
    logger.debug(
        "Verification results: entity_coverage=%.2f, embedding_sim=%.2f, "
        "entities_verified=%s, embeddings_verified=%s, overall=%s",
        entity_cov,
        embed_sim,
        is_verified_entity,
        is_verified_embed,
        overall_verified,
    )

    return overall_verified, entity_cov, embed_sim, missing_entities


def calculate_hallucination_score(
    entity_coverage: float, embedding_similarity: Optional[float] = None, entity_weight: float = 0.6
) -> float:
    """Calculate a combined hallucination score (0-1 scale).

    Lower scores indicate higher likelihood of hallucination.

    Args:
        entity_coverage: Ratio of entities found in context.
        embedding_similarity: Embedding similarity score (if available).
        entity_weight: Weight given to entity coverage vs embedding.

    Returns:
        Combined score between 0-1, where lower values indicate possible
        hallucinations.

    """
    if embedding_similarity is None:
        return entity_coverage

    # Combine scores with configurable weighting
    return entity_coverage * entity_weight + embedding_similarity * (1 - entity_weight)


def needs_human_review(
    hallucination_score: float,
    config: Optional[HallucinationConfig] = None,
    critical_threshold: Optional[float] = None,
    entity_coverage: float = 0.0,
    entity_critical_threshold: Optional[float] = None,
    embedding_similarity: Optional[float] = None,
    embedding_critical_threshold: Optional[float] = None,
) -> bool:
    """Determine if a response requires human review.

    Args:
        hallucination_score: Combined hallucination score.
        config: HallucinationConfig object with thresholds.
        critical_threshold: Override for the combined score threshold.
        entity_coverage: Entity coverage ratio.
        entity_critical_threshold: Override for entity critical threshold.
        embedding_similarity: Embedding similarity score.
        embedding_critical_threshold: Override for embedding threshold.

    Returns:
        True if human review is recommended.

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

    if embedding_similarity is not None and embedding_similarity < emb_threshold:
        return True

    return False


def generate_verification_warning(
    missing_entities: List[str], coverage_ratio: float, embed_sim: Optional[float] = None, human_review: bool = False
) -> str:
    """Generate a warning message for potentially hallucinated content.

    Args:
        missing_entities: List of entities not found in the context
        coverage_ratio: The ratio of entities that were found in the context
        embed_sim: Optional embedding similarity score
        human_review: Whether this response has been flagged for human review

    Returns:
        A warning message

    """
    if not missing_entities:
        return ""

    confidence_level = "low" if coverage_ratio < 0.5 else "moderate"

    warning = (
        "\n\n[SYSTEM WARNING: This response may contain hallucinated "
        f"information. Confidence level: {confidence_level}. "
        f"The following terms were not found in the retrieved documents: "
        f"{', '.join(missing_entities[:5])}"
    )

    if len(missing_entities) > 5:
        warning += f" and {len(missing_entities) - 5} more"

    if embed_sim is not None:
        warning += f". Semantic similarity score: {embed_sim:.2f}"

    if human_review:
        warning += ". This response has been flagged for expert review"

    warning += ". Please verify this information from other sources.]"

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
    verified, entity_cov, embed_sim, missing_entities = advanced_verify_response(
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
    hallucination_score = calculate_hallucination_score(entity_cov, embed_sim, config.entity_weight)

    # Check if human review is needed
    requires_review = False
    if config.flag_for_human_review:
        requires_review = needs_human_review(
            hallucination_score=hallucination_score,
            config=config,
            entity_coverage=entity_cov,
            embedding_similarity=embed_sim,
        )

    # Add warning if not verified or requires review
    if not verified or requires_review:
        warning = generate_verification_warning(missing_entities, entity_cov, embed_sim, requires_review)
        response += warning

    # Create metadata for caller
    metadata = {
        "verified": verified,
        "entity_coverage": entity_cov,
        "embedding_similarity": embed_sim,
        "missing_entities": missing_entities,
        "hallucination_score": hallucination_score,
        "human_review_recommended": requires_review,
    }

    if return_metadata:
        return response, metadata
    else:
        return response
