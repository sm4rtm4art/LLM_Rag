"""Module for comparing document sections using embeddings."""

from typing import List, Optional

import numpy as np
from scipy.spatial.distance import cosine

from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

from .component_protocols import IComparisonEngine
from .domain_models import (
    AlignmentPair,
    ComparisonConfig,
    ComparisonResultType,
    Section,
    SectionComparison,
)

logger = get_logger(__name__)


class EmbeddingComparisonEngine(IComparisonEngine):
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
            f'Initialized EmbeddingComparisonEngine with similar_threshold='
            f'{self.config.similarity_thresholds.similar}, '
            f'embedding_model={self.config.embedding_model_name}'
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
            logger.debug(f'Comparing {len(aligned_pairs)} aligned section pairs')
            comparisons = []

            for pair in aligned_pairs:
                if pair.is_source_only:
                    comparisons.append(
                        SectionComparison(
                            result_type=ComparisonResultType.DELETED, alignment_pair=pair, similarity_score=0.0
                        )
                    )
                elif pair.is_target_only:
                    comparisons.append(
                        SectionComparison(
                            result_type=ComparisonResultType.NEW, alignment_pair=pair, similarity_score=0.0
                        )
                    )
                else:
                    if pair.source_section and pair.target_section:
                        similarity = self._calculate_similarity(pair.source_section, pair.target_section)
                        result_classification = self._classify_similarity(similarity)
                        comparisons.append(
                            SectionComparison(
                                result_type=result_classification, alignment_pair=pair, similarity_score=similarity
                            )
                        )
                    else:
                        logger.warning(f'Skipping comparison for pair with missing section(s) in aligned pair: {pair}')

            logger.debug(f'Completed {len(comparisons)} section comparisons')
            return comparisons

        except Exception as e:
            error_msg = f'Error comparing document sections: {str(e)}'
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
        logger.debug(
            f'Calculating section similarity for source: {source_section.section_id}, '
            f'target: {target_section.section_id}'
        )
        source_embedding = self._get_embedding(source_section.content)
        target_embedding = self._get_embedding(target_section.content)
        similarity = (
            1.0 - cosine(source_embedding, target_embedding)
            if source_embedding is not None and target_embedding is not None
            else 0.0
        )
        logger.debug(f'Calculated similarity: {similarity:.4f}')
        return similarity

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a text string.

        Uses a caching mechanism to avoid recomputing embeddings.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector as numpy array.

        """
        if not text:
            return None
        cache_key = text[:100] + str(len(text))
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        if self._embedding_model is None:
            self._initialize_embedding_model()

        try:
            embedding = self._compute_embedding(text)
            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error computing embedding for text snippet '{text[:50]}...': {e}")
            return None

    def _initialize_embedding_model(self):
        """Initialize the embedding model.

        This is a placeholder that would be replaced with actual model initialization
        in a production implementation.
        """
        logger.info(f'Initializing embedding model: {self.config.embedding_model_name}')

        # In a real implementation, this would load a model like:
        # from sentence_transformers import SentenceTransformer
        # self._embedding_model = SentenceTransformer(self.config.embedding_model_name)
        # or integrate with existing RAG embedding infrastructure

        # For this implementation, we'll use a simple mock
        self._embedding_model = 'mock_model_initialized'
        logger.info('Embedding model initialized')

    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding for a text string.

        This is a simplified placeholder implementation. In a real system,
        this would use a proper embedding model like SentenceTransformer
        or integrate with the existing embedding infrastructure.

        Args:
            text: Text to compute embedding for.

        Returns:
            Embedding vector as numpy array.

        """
        if not text or not self._embedding_model:
            return None
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
        hash_obj = hashlib.md5(text.encode('utf-8'), usedforsecurity=False)
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

    def _classify_similarity(self, similarity: float) -> ComparisonResultType:
        """Classify the similarity score into a comparison result.

        Args:
            similarity: Similarity score between 0 and 1.

        Returns:
            Classification of the comparison result.

        """
        thresholds = self.config.similarity_thresholds
        if similarity >= thresholds.similar:
            return ComparisonResultType.SIMILAR
        elif similarity >= thresholds.modified:
            return ComparisonResultType.MODIFIED
        elif similarity >= thresholds.different:
            return ComparisonResultType.DIFFERENT
        else:
            return ComparisonResultType.DIFFERENT
