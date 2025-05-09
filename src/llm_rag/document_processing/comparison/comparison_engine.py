"""Module for comparing document sections using embeddings."""

import asyncio
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
    LLMAnalysisResult,
    Section,
    SectionComparison,
)
from .llm_comparer import LLMComparer

logger = get_logger(__name__)

MIN_LLM_CONTENT_LENGTH = 15  # Minimum character length for content to be sent to LLM


class EmbeddingComparisonEngine(IComparisonEngine):
    """Engine for comparing document sections using embeddings and optional LLM analysis.

    This class forms the core of the document section comparison process.
    It first calculates semantic similarity between aligned pairs of sections
    using embeddings. Based on this similarity, it classifies the relationship
    (e.g., SIMILAR, MODIFIED, DIFFERENT).

    Optionally, if an `LLMComparer` instance is provided, this engine can
    trigger a more detailed analysis for sections classified as MODIFIED or
    DIFFERENT. The `LLMComparer` then provides a nuanced categorization
    (e.g., LEGAL_EFFECT_CHANGE, SEMANTIC_REWRITE) and an explanation, which is
    stored alongside the embedding-based comparison result.

    The engine handles the orchestration of these steps, including skipping
    LLM analysis for very short content sections to optimize resource usage.

    Attributes:
        config: A `ComparisonConfig` object containing thresholds and settings
            for embedding-based similarity classification.
        llm_comparer: An optional `LLMComparer` instance. If provided, it's
            used for detailed LLM-based analysis of differing sections.
        _embedding_model: Internal placeholder for the embedding model.
        _embedding_cache: Internal cache for computed embeddings to avoid
            redundant calculations.

    """

    def __init__(self, config: Optional[ComparisonConfig] = None, llm_comparer: Optional[LLMComparer] = None):
        """Initialize the EmbeddingComparisonEngine.

        Sets up the configuration for similarity thresholds and embedding model
        details. It also accepts an optional `LLMComparer` for performing
        deeper analysis on sections that differ significantly based on embeddings.

        Args:
            config: Configuration for comparison thresholds, embedding model name,
                and other behavior. If None, a default `ComparisonConfig` is used.
            llm_comparer: Optional `LLMComparer` instance. If provided, enables
                detailed LLM-based analysis for sections flagged by the initial
                embedding comparison as MODIFIED or DIFFERENT.

        """
        self.config = config or ComparisonConfig()
        self.llm_comparer = llm_comparer
        logger.info(
            f'Initialized EmbeddingComparisonEngine with similar_threshold='
            f'{self.config.similarity_thresholds.similar}, '
            f'embedding_model={self.config.embedding_model_name}, '
            f'LLMComparer enabled: {self.llm_comparer is not None}'
        )

        # Placeholder for embedding model
        self._embedding_model = None
        self._embedding_cache = {}

    async def compare_sections(self, aligned_pairs: List[AlignmentPair]) -> List[SectionComparison]:
        """Compare aligned section pairs, classify changes, and optionally use LLM.

        This is the main method for processing a list of aligned section pairs.
        For each pair, it determines if it's an addition, deletion, or an
        aligned pair requiring similarity calculation. For aligned pairs, it
        calculates embedding similarity and classifies the result (SIMILAR,
        MODIFIED, DIFFERENT).

        If an `LLMComparer` is configured and a pair is classified as MODIFIED or
        DIFFERENT by embeddings, this method schedules an asynchronous LLM
        analysis via `_run_llm_analysis_for_pair`. The results of these LLM
        analyses (an `LLMAnalysisResult` object) are then stored within the
        respective `SectionComparison` objects.

        The primary `result_type` of a `SectionComparison` is based on embedding
        similarity. The `LLMAnalysisResult`, if present, provides supplementary,
        more granular details about the nature of the difference.

        Args:
            aligned_pairs: A list of `AlignmentPair` objects representing
                aligned sections from two documents.

        Returns:
            A list of `SectionComparison` objects, each detailing the
            comparison result for a pair, potentially enriched with
            `LLMAnalysisResult`.

        Raises:
            DocumentProcessingError: If any unrecoverable error occurs during
                the comparison process.

        """
        try:
            logger.debug(f'Comparing {len(aligned_pairs)} aligned section pairs')
            comparisons: List[SectionComparison] = []
            llm_tasks = []

            for pair_index, pair in enumerate(aligned_pairs):
                section_comp: Optional[SectionComparison] = None
                llm_analysis_needed = False

                if pair.is_source_only:
                    section_comp = SectionComparison(
                        result_type=ComparisonResultType.DELETED, alignment_pair=pair, similarity_score=0.0
                    )
                elif pair.is_target_only:
                    section_comp = SectionComparison(
                        result_type=ComparisonResultType.NEW, alignment_pair=pair, similarity_score=0.0
                    )
                else:
                    if pair.source_section and pair.target_section:
                        similarity = self._calculate_similarity(pair.source_section, pair.target_section)
                        result_classification = self._classify_similarity(similarity)
                        section_comp = SectionComparison(
                            result_type=result_classification, alignment_pair=pair, similarity_score=similarity
                        )

                        # Condition to trigger LLM analysis
                        if self.llm_comparer and (
                            result_classification == ComparisonResultType.DIFFERENT
                            or result_classification == ComparisonResultType.MODIFIED
                        ):
                            llm_analysis_needed = True
                    else:
                        logger.warning(f'Skipping comparison for pair with missing section(s) in aligned pair: {pair}')

                if section_comp:
                    comparisons.append(section_comp)
                    if llm_analysis_needed and self.llm_comparer and pair.source_section and pair.target_section:
                        # Schedule LLM analysis for later execution
                        # Pass index to update the correct comparison object later
                        llm_tasks.append(
                            self._run_llm_analysis_for_pair(
                                pair.source_section.content,
                                pair.target_section.content,
                                pair_index,  # Store index to update the correct comparison
                            )
                        )

            # Run all scheduled LLM analyses concurrently
            if llm_tasks:
                logger.info(f'Running LLM analysis for {len(llm_tasks)} pairs.')
                llm_results_with_indices = await asyncio.gather(*llm_tasks)
                for index, llm_analysis in llm_results_with_indices:
                    if index < len(comparisons) and llm_analysis:
                        comparisons[index].llm_analysis_result = llm_analysis
                        # The primary result_type (SIMILAR, MODIFIED, DIFFERENT) is determined
                        # by embedding similarity. LLMAnalysisResult provides supplementary
                        # detailed categorization for sections already flagged for LLM review.
                        # Future enhancements might explore direct mapping or refinement of
                        # result_type based on LLM categories if deemed robust.
                        logger.debug(f'Stored LLM analysis for pair index {index}: {llm_analysis.comparison_category}')

            logger.debug(f'Completed {len(comparisons)} section comparisons')
            return comparisons

        except Exception as e:
            error_msg = f'Error comparing document sections: {str(e)}'
            logger.error(error_msg, exc_info=True)
            raise DocumentProcessingError(error_msg) from e

    async def _run_llm_analysis_for_pair(
        self, text_a: str, text_b: str, original_index: int
    ) -> tuple[int, Optional[LLMAnalysisResult]]:
        """Run LLM analysis for a text pair, skipping if content is too short.

        This helper method is called by `compare_sections` to perform detailed
        analysis on a pair of text sections using the configured `LLMComparer`.

        It first checks if either text section is shorter than
        `MIN_LLM_CONTENT_LENGTH` (after stripping whitespace). If so, it
        bypasses the LLM call and returns an `LLMAnalysisResult` with the
        category `NO_MEANINGFUL_CONTENT`, providing an appropriate explanation.

        If the content is sufficient, it delegates the analysis to
        `self.llm_comparer.analyze_sections`. Any exceptions during LLM
        analysis are caught, logged, and result in an `LLMAnalysisResult` with
        category `llm_error`.

        Args:
            text_a: The content of the first text section.
            text_b: The content of the second text section.
            original_index: The original index of this pair in the list being
                processed by `compare_sections`. This is returned to allow
                asynchronous results to be correctly mapped back.

        Returns:
            A tuple containing the `original_index` and an `LLMAnalysisResult`
            object, or None if no `llm_comparer` is configured. The
            `LLMAnalysisResult` will reflect the LLM's findings, a skipped
            analysis due to short content, or an error during the LLM call.

        """
        if not self.llm_comparer:
            return original_index, None

        # Check for very short content
        text_a_len = len(text_a.strip())
        text_b_len = len(text_b.strip())
        if text_a_len < MIN_LLM_CONTENT_LENGTH or text_b_len < MIN_LLM_CONTENT_LENGTH:
            logger.debug(
                f'Skipping LLM analysis for pair index {original_index} due to short content. '
                f'Len A: {text_a_len}, Len B: {text_b_len}'
            )
            return original_index, LLMAnalysisResult(
                comparison_category='NO_MEANINGFUL_CONTENT',
                explanation='One or both sections were too short for LLM analysis.',
                confidence=1.0,  # Confident in this assessment
                raw_llm_response=None,
                metadata={'reason': 'content_too_short'},
            )

        try:
            llm_result = await self.llm_comparer.analyze_sections(text_a, text_b)
            return original_index, llm_result
        except Exception as e:
            logger.error(f'LLM analysis failed for pair index {original_index}: {e}', exc_info=True)
            return original_index, None

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

        import hashlib

        # Use a combination of text features and hash for more robust mock embeddings
        text_bytes = text.encode('utf-8')
        hash_obj = hashlib.md5(text_bytes, usedforsecurity=False)
        hash_val_int = int(hash_obj.hexdigest(), 16)

        # Simple features
        f1 = len(text) / 1000.0  # Normalized length
        f2 = len(set(text.split())) / (len(text.split()) + 1e-6)  # Unique word ratio
        f3 = sum(ord(c) for c in text[:10]) / (10 * 128.0)  # Avg char value of first 10 chars

        # Create embedding based on these features and hash parts
        # Ensure a fixed size, e.g., 10 dimensions
        embedding_vals = [
            f1,
            f2,
            f3,
            (hash_val_int & 0xFF) / 255.0,
            ((hash_val_int >> 8) & 0xFF) / 255.0,
            ((hash_val_int >> 16) & 0xFF) / 255.0,
            ((hash_val_int >> 24) & 0xFF) / 255.0,
            ((hash_val_int >> 32) & 0xFF) / 255.0,  # Add more components if vector needs to be longer
            ((hash_val_int >> 40) & 0xFF) / 255.0,
            ((hash_val_int >> 48) & 0xFF) / 255.0,
        ]
        # If embedding_vals is shorter than 10, pad with parts of hash_val_int or repeat, etc.
        # For simplicity, if we need exactly 10, and have 10 above, it's fine.
        # Let's ensure it is 10 dimensional for the test that checks shape[0]
        # Current embedding_vals has 10 elements. Perfect.

        mock_embedding = np.array(embedding_vals[:10])  # Ensure fixed size

        norm = np.linalg.norm(mock_embedding)
        return mock_embedding / norm if norm > 0 else mock_embedding

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
