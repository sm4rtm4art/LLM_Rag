"""Module for comparing document sections using embeddings."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cosine

from llm_rag.document_processing.comparison.alignment import AlignmentPair
from llm_rag.document_processing.comparison.document_parser import Section
from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


class ComparisonResult(Enum):
    """Classification of comparison results."""

    SIMILAR = "similar"
    MINOR_CHANGES = "minor_changes"
    MAJOR_CHANGES = "major_changes"
    REWRITTEN = "rewritten"
    NEW = "new"
    DELETED = "deleted"


@dataclass
class SectionComparison:
    """Result of comparing two document sections.

    Attributes:
        alignment_pair: The aligned sections being compared.
        result: Classification of the comparison result.
        similarity_score: Numeric similarity score between 0 and 1.
        details: Additional information about the comparison.

    """

    alignment_pair: AlignmentPair
    result: ComparisonResult
    similarity_score: float
    details: Optional[Dict] = None


@dataclass
class ComparisonConfig:
    """Configuration settings for section comparison.

    Attributes:
        similar_threshold: Minimum similarity score to consider sections similar.
        minor_change_threshold: Minimum score for minor changes classification.
        major_change_threshold: Minimum score for major changes classification.
        rewritten_threshold: Minimum score to consider text rewritten.
        embedding_model: Name of the embedding model to use.
        chunk_size: Size of text chunks for embedding (if applicable).

    """

    similar_threshold: float = 0.9
    minor_change_threshold: float = 0.8
    major_change_threshold: float = 0.6
    rewritten_threshold: float = 0.4
    embedding_model: str = "default"
    chunk_size: int = 512


class EmbeddingComparisonEngine:
    """Engine for comparing document sections using embeddings.

    This class calculates embeddings for sections and uses them to
    determine similarity and classify changes.
    """

    def __init__(self, config: Optional[ComparisonConfig] = None):
        """Initialize the comparison engine.

        Args:
            config: Configuration for comparison thresholds and behavior.
                If None, default configuration will be used.

        """
        self.config = config or ComparisonConfig()
        logger.info(
            f"Initialized EmbeddingComparisonEngine with similar_threshold={self.config.similar_threshold}, "
            f"embedding_model={self.config.embedding_model}"
        )

        # Placeholder for embedding model
        self._embedding_model = None
        self._embedding_cache = {}

    def compare_sections(self, aligned_pairs: List[AlignmentPair]) -> List[SectionComparison]:
        """Compare aligned section pairs and classify the changes.

        Args:
            aligned_pairs: List of aligned section pairs to compare.

        Returns:
            List of comparison results with classifications.

        Raises:
            DocumentProcessingError: If comparison fails.

        """
        try:
            logger.debug(f"Comparing {len(aligned_pairs)} aligned section pairs")
            comparisons = []

            for pair in aligned_pairs:
                # Handle source-only and target-only cases
                if pair.is_source_only:
                    comparisons.append(
                        SectionComparison(alignment_pair=pair, result=ComparisonResult.DELETED, similarity_score=0.0)
                    )
                elif pair.is_target_only:
                    comparisons.append(
                        SectionComparison(alignment_pair=pair, result=ComparisonResult.NEW, similarity_score=0.0)
                    )
                else:
                    # Compare two aligned sections
                    if pair.similarity_score == 0.0:
                        # Calculate similarity if not already done during alignment
                        similarity = self._calculate_similarity(pair.source_section, pair.target_section)
                    else:
                        similarity = pair.similarity_score

                    # Classify the change based on similarity score
                    result = self._classify_similarity(similarity)

                    comparisons.append(
                        SectionComparison(alignment_pair=pair, result=result, similarity_score=similarity)
                    )

            logger.debug(f"Completed {len(comparisons)} section comparisons")
            return comparisons

        except Exception as e:
            error_msg = f"Error comparing document sections: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def _calculate_similarity(self, source_section: Section, target_section: Section) -> float:
        """Calculate similarity between two sections using embeddings.

        Args:
            source_section: Source document section.
            target_section: Target document section.

        Returns:
            Similarity score between 0 and 1.

        """
        logger.debug("Calculating section similarity using embeddings")

        # Get or compute embeddings
        source_embedding = self._get_embedding(source_section.content)
        target_embedding = self._get_embedding(target_section.content)

        # Calculate cosine similarity
        similarity = 1.0 - cosine(source_embedding, target_embedding)

        logger.debug(f"Calculated similarity: {similarity:.4f}")
        return similarity

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string.

        Uses a caching mechanism to avoid recomputing embeddings.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as numpy array.

        """
        # Check cache first
        cache_key = text[:100]  # Use first 100 chars as key for memory efficiency
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Initialize embedding model if not already done
        if self._embedding_model is None:
            self._initialize_embedding_model()

        # Compute embedding
        embedding = self._compute_embedding(text)

        # Cache result
        self._embedding_cache[cache_key] = embedding
        return embedding

    def _initialize_embedding_model(self):
        """Initialize the embedding model.

        This is a placeholder that would be replaced with actual model initialization
        in a production implementation.
        """
        logger.info(f"Initializing embedding model: {self.config.embedding_model}")

        # In a real implementation, this would load a model like:
        # self._embedding_model = SentenceTransformer(self.config.embedding_model)
        # or integrate with existing RAG embedding infrastructure

        # For this implementation, we'll use a simple mock
        self._embedding_model = "mock_model"
        logger.info("Embedding model initialized")

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a text string.

        This is a simplified placeholder implementation. In a real system,
        this would use a proper embedding model like SentenceTransformer
        or integrate with the existing embedding infrastructure.

        Args:
            text: Text to compute embedding for.

        Returns:
            Embedding vector as numpy array.

        """
        # For demonstration purposes, we'll create a simple mock embedding
        # In a real implementation, this would be:
        # embedding = self._embedding_model.encode(text)

        # Create a deterministic but simplified mock embedding based on text features
        words = text.lower().split()
        unique_words = set(words)

        # Simple features:
        # - Normalized text length
        # - Unique word ratio
        # - Average word length

        text_length = min(1.0, len(text) / 1000)
        unique_ratio = len(unique_words) / max(1, len(words))
        avg_word_len = sum(len(w) for w in words) / max(1, len(words)) / 10

        # Create a mock 10-dimensional embedding with some randomness but
        # deterministic for the same input
        import hashlib

        # Get a hash-based seed for reproducibility
        hash_obj = hashlib.md5(text.encode("utf-8"), usedforsecurity=False)
        hash_val = int(hash_obj.hexdigest(), 16)

        # Ensure hash_val is within valid range for numpy's random seed (0 to 2^32-1)
        hash_val = hash_val % (2**32)

        # Create a mock embedding with some simple features and some hash-based values
        np.random.seed(hash_val)
        random_vals = np.random.rand(7)

        mock_embedding = np.array([text_length, unique_ratio, avg_word_len, *random_vals])

        # Normalize embedding
        norm = np.linalg.norm(mock_embedding)
        if norm > 0:
            mock_embedding = mock_embedding / norm

        return mock_embedding

    def _classify_similarity(self, similarity: float) -> ComparisonResult:
        """Classify the similarity score into a comparison result.

        Args:
            similarity: Similarity score between 0 and 1.

        Returns:
            Classification of the comparison result.

        """
        if similarity >= self.config.similar_threshold:
            return ComparisonResult.SIMILAR
        elif similarity >= self.config.minor_change_threshold:
            return ComparisonResult.MINOR_CHANGES
        elif similarity >= self.config.major_change_threshold:
            return ComparisonResult.MAJOR_CHANGES
        elif similarity >= self.config.rewritten_threshold:
            return ComparisonResult.REWRITTEN
        else:
            # Very low similarity could indicate completely different content,
            # but the alignment algorithm should have handled this with NEW/DELETED
            return ComparisonResult.MAJOR_CHANGES
