"""Configuration for hallucination detection and verification.

This module provides configuration classes for customizing the behavior of the
anti-hallucination module.
"""

from dataclasses import dataclass


@dataclass
class HallucinationConfig:
    """Configuration for hallucination detection and verification.

    This class encapsulates all configuration parameters for entity-based
    and embedding-based verification, as well as human review thresholds.

    Attributes:
        entity_threshold: Minimum ratio of entities that must be found in context (0-1)
        embedding_threshold: Minimum cosine similarity score required (0-1)
        model_name: Name of the sentence transformer model to use
        entity_weight: Weight given to entity verification vs embedding (0-1)
        human_review_threshold: Overall score below which human review is recommended
        entity_critical_threshold: Entity coverage below which review is critical
        embedding_critical_threshold: Similarity score below which review is critical
        use_embeddings: Whether to use embedding-based verification
        flag_for_human_review: Whether to flag responses for human review

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
