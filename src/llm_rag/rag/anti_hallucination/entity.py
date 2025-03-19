"""Entity-based verification for anti-hallucination.

This module provides functions for extracting entities from text and verifying
that entities in a response are present in the context.
"""

import re
from typing import List, Set, Tuple

# Local imports
from llm_rag.rag.anti_hallucination.utils import load_stopwords


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
    """Verify that entities in the response are present in the context.

    Args:
        response: The generated response
        context: The context used for generation
        threshold: Minimum required ratio of covered entities (0-1)
        languages: List of languages to use for stopword removal

    Returns:
        Tuple containing:
            - Verification success flag (bool)
            - Coverage ratio (float)
            - List of missing entities (List[str])

    """
    # Get entities from both response and context
    response_entities = extract_key_entities(response, languages)
    context_entities = extract_key_entities(context, languages)

    if not response_entities:
        # No entities in response, can't verify
        return True, 1.0, []

    # Find uncovered entities
    missing_entities = []
    for entity in response_entities:
        if entity not in context_entities:
            missing_entities.append(entity)

    # Calculate ratio of covered entities
    coverage_ratio = 1 - (len(missing_entities) / len(response_entities))

    # Determine if verification passed
    verified = coverage_ratio >= threshold

    return verified, coverage_ratio, missing_entities
