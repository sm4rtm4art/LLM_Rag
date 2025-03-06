"""Anti-hallucination utilities for RAG systems.

This module provides utilities to reduce hallucinations in RAG systems.
"""

import logging
import re
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)


def extract_key_entities(text: str) -> Set[str]:
    """Extract key entities from text.

    Args:
        text: The text to extract entities from

    Returns:
        A set of key entities

    """
    # Remove common stop words (English and German)
    stop_words = {
        # English stop words
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
        # German stopwords
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
    }

    # Extract words, filter out stop words and short words
    # Include German special characters (ä, ö, ü, ß) in the pattern
    words = re.findall(r"\b[a-zA-ZäöüßÄÖÜ0-9_-]{4,}\b", text.lower())
    entities = {word for word in words if word not in stop_words}

    # Add special handling for DIN standard references
    din_pattern = r"\b(?:DIN|EN|ISO|IEC|CEN|TR)\s*[-_]?\s*\d+(?:[-_]\d+)*\b"
    din_refs = re.findall(din_pattern, text)
    entities.update([ref.replace(" ", "").lower() for ref in din_refs])

    return entities


def verify_entities_in_context(response: str, context: str, threshold: float = 0.7) -> Tuple[bool, float, List[str]]:
    """Verify that entities in the response are present in the context.

    Args:
        response: The generated response
        context: The context used to generate the response
        threshold: The minimum ratio of entities that must be in the context

    Returns:
        A tuple of (is_verified, entity_coverage_ratio, missing_entities)

    """
    # Extract entities from response and context
    response_entities = extract_key_entities(response)
    context_entities = extract_key_entities(context)

    if not response_entities:
        return True, 1.0, []

    # Find entities in response that are not in context
    missing_entities = [entity for entity in response_entities if entity not in context_entities]

    # Calculate coverage ratio
    coverage_ratio = 1.0 - (len(missing_entities) / len(response_entities))

    # Determine if the response is verified
    is_verified = coverage_ratio >= threshold

    return is_verified, coverage_ratio, missing_entities


def generate_verification_warning(missing_entities: List[str], coverage_ratio: float) -> str:
    """Generate a warning message for potentially hallucinated content.

    Args:
        missing_entities: List of entities not found in the context
        coverage_ratio: The ratio of entities that were found in the context

    Returns:
        A warning message

    """
    if not missing_entities:
        return ""

    confidence_level = "low" if coverage_ratio < 0.5 else "moderate"

    warning = (
        "\n\n[SYSTEM WARNING: This response may contain hallucinated information. "
        f"Confidence level: {confidence_level}. The following terms were not found "
        f"in the retrieved documents: {', '.join(missing_entities[:5])}"
    )

    if len(missing_entities) > 5:
        warning += f" and {len(missing_entities) - 5} more"

    warning += ". Please verify this information from other sources.]"

    return warning


def post_process_response(response: str, context: str, threshold: float = 0.7) -> str:
    """Post-process a response to add warnings about potential hallucinations.

    Args:
        response: The generated response
        context: The context used to generate the response
        threshold: The minimum ratio of entities that must be in the context

    Returns:
        The response with warnings added if necessary

    """
    is_verified, coverage_ratio, missing_entities = verify_entities_in_context(response, context, threshold)

    if not is_verified:
        warning = generate_verification_warning(missing_entities, coverage_ratio)
        response += warning

    return response
