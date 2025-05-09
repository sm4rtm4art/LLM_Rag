"""Module for orchestrating document comparison workflow."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

# Import component classes
from .alignment import SectionAligner
from .comparison_engine import EmbeddingComparisonEngine
from .diff_formatter import DiffFormatter
from .document_parser import DocumentParser

# Import ALL data models and Configs from domain_models
from .domain_models import (
    ComparisonPipelineConfig,
    DocumentFormat,
    Section,
    SectionComparison,
)

logger = get_logger(__name__)


class ComparisonPipeline:
    """Pipeline for comparing documents.

    This class orchestrates the entire comparison workflow, from parsing
    documents to generating a diff report.
    """

    def __init__(self, config: Optional[ComparisonPipelineConfig] = None):
        """Initialize the comparison pipeline.

        Args:
            config: Configuration for the pipeline.
                If None, default configuration will be used.

        """
        self.config = config or ComparisonPipelineConfig()
        logger.info(
            f'Initialized ComparisonPipeline with cache_intermediate_results={self.config.cache_intermediate_results}'
        )

        # Initialize components using their respective configs from the pipeline config
        self.parser = DocumentParser(self.config.parser_config)
        self.aligner = SectionAligner(self.config.alignment_config)
        self.comparison_engine = EmbeddingComparisonEngine(self.config.comparison_config)
        self.formatter = DiffFormatter(self.config.formatter_config)

        # Cache for intermediate results
        self._cache = {}

    def compare_documents(
        self,
        source_document: Union[str, Path],
        target_document: Union[str, Path],
        source_format: DocumentFormat = DocumentFormat.MARKDOWN,
        target_format: DocumentFormat = DocumentFormat.MARKDOWN,
        title: Optional[str] = None,
    ) -> str:
        """Compare two documents and generate a diff report.

        Args:
            source_document: Source document content or path.
            target_document: Target document content or path.
            source_format: Format of the source document.
            target_format: Format of the target document.
            title: Optional title for the diff report.

        Returns:
            Formatted diff report as a string.

        Raises:
            DocumentProcessingError: If comparison fails.

        """
        try:
            logger.info('Starting document comparison')

            # Parse documents
            source_sections = self._parse_document(source_document, source_format)
            target_sections = self._parse_document(target_document, target_format)

            # Align sections
            aligned_pairs = self._align_sections(source_sections, target_sections)

            # Compare sections
            comparison_results = self._compare_sections(aligned_pairs)

            # Format results
            diff_report = self._format_results(comparison_results, title)

            logger.info('Document comparison completed successfully')
            return diff_report

        except Exception as e:
            error_msg = f'Error in document comparison pipeline: {str(e)}'
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def compare_sections(
        self,
        source_sections: List[Section],
        target_sections: List[Section],
        title: Optional[str] = None,
    ) -> str:
        """Compare pre-parsed document sections.

        Args:
            source_sections: List of sections from the source document.
            target_sections: List of sections from the target document.
            title: Optional title for the diff report.

        Returns:
            Formatted diff report as a string.

        Raises:
            DocumentProcessingError: If comparison fails.

        """
        try:
            logger.info('Starting section comparison')

            # Align sections
            aligned_pairs = self._align_sections(source_sections, target_sections)

            # Compare sections
            comparison_results = self._compare_sections(aligned_pairs)

            # Format results
            diff_report = self._format_results(comparison_results, title)

            logger.info('Section comparison completed successfully')
            return diff_report

        except Exception as e:
            error_msg = f'Error in section comparison: {str(e)}'
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def _load_document(self, document: Union[str, Path]) -> str:
        """Load a document from a file path.

        Args:
            document: Document path or content.

        Returns:
            Document content as a string.

        Raises:
            DocumentProcessingError: If document loading fails.

        """
        try:
            # Check if the document is a path
            is_path = isinstance(document, Path)
            is_str_path = (
                isinstance(document, str)
                and not document.startswith(('# ', '{', '<', '---'))
                and '\n' not in document[:100]
            )

            path_exists = False
            if is_str_path:
                path_exists = Path(document).exists()

            is_likely_path = is_path or (is_str_path and path_exists)

            if is_likely_path:
                document_path = Path(document)
                logger.debug(f'Loading document from path: {document_path}')
                return document_path.read_text(encoding='utf-8')

            # Assume it's already content
            logger.debug('Using provided document content')
            return str(document)

        except Exception as e:
            error_msg = f'Error loading document: {str(e)}'
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def _parse_document(self, document: Union[str, Path], format_type: DocumentFormat) -> List[Section]:
        """Parse a document into sections.

        Args:
            document: Document content or path.
            format_type: Format of the document.

        Returns:
            List of document sections.

        """
        cache_key = f'parsed_{document}'
        if self.config.cache_intermediate_results and cache_key in self._cache:
            logger.debug(f'Using cached parsed document: {cache_key}')
            return self._cache[cache_key]

        # Load document if it's a path
        document_content = self._load_document(document)

        # Parse document
        logger.debug(f'Parsing document with format: {format_type.value}')
        sections = self.parser.parse(document_content, format_type)

        if self.config.cache_intermediate_results:
            self._cache[cache_key] = sections

        return sections

    def _align_sections(
        self, source_sections: List[Section], target_sections: List[Section]
    ) -> List[Tuple[Optional[Section], Optional[Section]]]:
        """Align sections between source and target documents.

        Args:
            source_sections: List of sections from the source document.
            target_sections: List of sections from the target document.

        Returns:
            List of alignment pairs.

        """
        cache_key = f'aligned_{id(source_sections)}_{id(target_sections)}'
        if self.config.cache_intermediate_results and cache_key in self._cache:
            logger.debug(f'Using cached section alignment: {cache_key}')
            return self._cache[cache_key]

        logger.debug('Aligning document sections')
        aligned_pairs = self.aligner.align_sections(source_sections, target_sections)

        if self.config.cache_intermediate_results:
            self._cache[cache_key] = aligned_pairs

        return aligned_pairs

    def _compare_sections(
        self,
        aligned_pairs: List[Tuple[Optional[Section], Optional[Section]]],
    ) -> List[SectionComparison]:
        """Compare aligned section pairs.

        Args:
            aligned_pairs: List of alignment pairs.

        Returns:
            List of section comparisons.

        """
        cache_key = f'compared_{id(aligned_pairs)}'
        if self.config.cache_intermediate_results and cache_key in self._cache:
            logger.debug(f'Using cached section comparisons: {cache_key}')
            return self._cache[cache_key]

        logger.debug('Comparing aligned section pairs')
        comparisons = self.comparison_engine.compare_sections(aligned_pairs)

        if self.config.cache_intermediate_results:
            self._cache[cache_key] = comparisons

        return comparisons

    def _format_results(
        self,
        comparisons: List[SectionComparison],
        title: Optional[str] = None,
    ) -> str:
        """Format comparison results.

        Args:
            comparisons: List of section comparisons.
            title: Optional title for the diff report.

        Returns:
            Formatted diff report as a string.

        """
        logger.debug('Formatting comparison results')
        return self.formatter.format_comparisons(comparisons, title)
