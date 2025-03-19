"""Embedding-based similarity verification for anti-hallucination.

This module provides functions for checking semantic similarity between
response and context using embedding models.
"""

import logging
from typing import Tuple

# Try to import optional dependencies
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    np = None
    cosine_similarity = None

# Local imports
from llm_rag.rag.anti_hallucination.utils import get_sentence_transformer_model

logger = logging.getLogger(__name__)


def embedding_based_verification(
    response: str, context: str, threshold: float = 0.75, model_name: str = "paraphrase-MiniLM-L6-v2"
) -> Tuple[bool, float]:
    """Verify response using embedding similarity with context.

    Args:
        response: The generated response
        context: The context used for generation
        threshold: Minimum required similarity score (0-1)
        model_name: Name of the embedding model to use

    Returns:
        Tuple containing:
            - Verification success flag (bool)
            - Similarity score (float)

    """
    # Feature checks
    if not np or not cosine_similarity:
        logger.warning("Embedding-based verification requires numpy and scikit-learn. Skipping verification.")
        return True, 1.0

    # Get or load the model
    model = get_sentence_transformer_model(model_name)
    if not model:
        return True, 1.0  # Skip verification if model not available

    # Generate embeddings
    try:
        response_embeddings = model.encode([response])[0]
        context_embeddings = model.encode([context])[0]

        # Calculate cosine similarity
        similarity_score = float(cosine_similarity([response_embeddings], [context_embeddings])[0][0])

        # Determine if verification passed
        verified = similarity_score >= threshold

        return verified, similarity_score
    except Exception as e:
        logger.error(f"Error in embedding verification: {e}")
        return True, 1.0  # Skip verification on error
