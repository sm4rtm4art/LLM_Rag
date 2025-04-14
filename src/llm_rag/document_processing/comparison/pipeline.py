"""Module for orchestrating the document comparison workflow."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from llm_rag.document_processing.comparison.alignment import AlignmentConfig, AlignmentMethod, SectionAligner
from llm_rag.document_processing.comparison.comparison_engine import ComparisonConfig, EmbeddingComparisonEngine
from llm_rag.document_processing.comparison.diff_formatter import DiffFormatter, FormatStyle, FormatterConfig
from llm_rag.document_processing.comparison.document_parser import DocumentFormat, DocumentParser, Section
from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonPipelineConfig:
    """Configuration for the document comparison pipeline.

    This class encapsulates all configuration options for the document
    comparison workflow.
    """

    # Document parser config
    default_document_format: DocumentFormat = DocumentFormat.MARKDOWN

    # Alignment config
    alignment_method: AlignmentMethod = AlignmentMethod.HYBRID
    similarity_threshold: float = 0.7
    use_sequence_information: bool = True
    heading_weight: float = 2.0
    content_weight: float = 1.0

    # Comparison engine config
    similar_threshold: float = 0.9
    minor_change_threshold: float = 0.8
    major_change_threshold: float = 0.6
    rewritten_threshold: float = 0.4
    embedding_model: str = "default"

    # Output formatter config
    output_format: FormatStyle = FormatStyle.MARKDOWN
    show_similarity_scores: bool = True
    detail_level: int = 2
    color_output: bool = True


class ComparisonPipeline:
    """Pipeline for comparing documents.

    This class orchestrates the complete document comparison workflow:
    1. Parsing documents into sections
    2. Aligning corresponding sections
    3. Comparing aligned sections
    4. Formatting the comparison results
    """

    def __init__(self, config: Optional[ComparisonPipelineConfig] = None):
        """Initialize the document comparison pipeline.

        Args:
            config: Configuration for the pipeline. If None, defaults are used.

        """
        self.config = config or ComparisonPipelineConfig()
        logger.info("Initializing document comparison pipeline")

        # Initialize components
        self._init_components()

        logger.info("Document comparison pipeline initialized successfully")

    def _init_components(self) -> None:
        """Initialize the pipeline components with the appropriate configuration."""
        # Document parser
        self.parser = DocumentParser(default_format=self.config.default_document_format)

        # Section aligner
        alignment_config = AlignmentConfig(
            method=self.config.alignment_method,
            similarity_threshold=self.config.similarity_threshold,
            use_sequence_information=self.config.use_sequence_information,
            heading_weight=self.config.heading_weight,
            content_weight=self.config.content_weight,
        )
        self.aligner = SectionAligner(config=alignment_config)

        # Comparison engine
        comparison_config = ComparisonConfig(
            similar_threshold=self.config.similar_threshold,
            minor_change_threshold=self.config.minor_change_threshold,
            major_change_threshold=self.config.major_change_threshold,
            rewritten_threshold=self.config.rewritten_threshold,
            embedding_model=self.config.embedding_model,
        )
        self.comparison_engine = EmbeddingComparisonEngine(config=comparison_config)

        # Diff formatter
        formatter_config = FormatterConfig(
            style=self.config.output_format,
            show_similarity_scores=self.config.show_similarity_scores,
            detail_level=self.config.detail_level,
            color_output=self.config.color_output,
        )
        self.formatter = DiffFormatter(config=formatter_config)

    def compare_documents(
        self,
        source_document: Union[str, Path],
        target_document: Union[str, Path],
        source_format: Optional[DocumentFormat] = None,
        target_format: Optional[DocumentFormat] = None,
        output_title: Optional[str] = None,
    ) -> str:
        """Compare two documents and generate a diff report.

        Args:
            source_document: Source document content or file path.
            target_document: Target document content or file path.
            source_format: Format of the source document (optional).
            target_format: Format of the target document (optional).
            output_title: Title for the diff report.

        Returns:
            Formatted comparison report.

        Raises:
            DocumentProcessingError: If comparison fails at any stage.

        """
        try:
            logger.info("Starting document comparison")

            # Parse documents
            logger.debug("Parsing source document")
            source_sections = self.parser.parse(source_document, format=source_format)
            logger.debug(f"Parsed {len(source_sections)} sections from source document")

            logger.debug("Parsing target document")
            target_sections = self.parser.parse(target_document, format=target_format)
            logger.debug(f"Parsed {len(target_sections)} sections from target document")

            # Align sections
            logger.debug("Aligning document sections")
            aligned_pairs = self.aligner.align_sections(source_sections, target_sections)
            logger.debug(f"Created {len(aligned_pairs)} section alignments")

            # Compare sections
            logger.debug("Comparing aligned sections")
            section_comparisons = self.comparison_engine.compare_sections(aligned_pairs)
            logger.debug(f"Completed {len(section_comparisons)} section comparisons")

            # Format output
            logger.debug("Formatting comparison results")
            report = self.formatter.format_comparisons(section_comparisons, title=output_title)

            logger.info(f"Document comparison completed successfully, generated {len(report)} character report")
            return report

        except Exception as e:
            error_msg = f"Error comparing documents: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def compare_document_sections(
        self, source_sections: List[Section], target_sections: List[Section], output_title: Optional[str] = None
    ) -> str:
        """Compare pre-parsed document sections.

        This is useful when integrating with other document processing workflows.

        Args:
            source_sections: Parsed source document sections.
            target_sections: Parsed target document sections.
            output_title: Title for the diff report.

        Returns:
            Formatted comparison report.

        Raises:
            DocumentProcessingError: If comparison fails at any stage.

        """
        try:
            logger.info("Starting comparison of pre-parsed document sections")
            logger.debug(f"Source: {len(source_sections)} sections, Target: {len(target_sections)} sections")

            # Align sections
            aligned_pairs = self.aligner.align_sections(source_sections, target_sections)
            logger.debug(f"Created {len(aligned_pairs)} section alignments")

            # Compare sections
            section_comparisons = self.comparison_engine.compare_sections(aligned_pairs)
            logger.debug(f"Completed {len(section_comparisons)} section comparisons")

            # Format output
            report = self.formatter.format_comparisons(section_comparisons, title=output_title)

            logger.info(f"Section comparison completed successfully, generated {len(report)} character report")
            return report

        except Exception as e:
            error_msg = f"Error comparing document sections: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e

    def save_comparison_report(self, report: str, output_path: Union[str, Path], overwrite: bool = False) -> None:
        """Save the comparison report to a file.

        Args:
            report: The comparison report to save.
            output_path: Path where the report should be saved.
            overwrite: Whether to overwrite existing files.

        Raises:
            DocumentProcessingError: If the report cannot be saved.

        """
        try:
            path = Path(output_path)
            logger.debug(f"Saving comparison report to {path}")

            if path.exists() and not overwrite:
                raise ValueError(f"Output file already exists: {path}. Use overwrite=True to replace.")

            with open(path, "w", encoding="utf-8") as f:
                f.write(report)

            logger.info(f"Comparison report saved to {path}")

        except Exception as e:
            error_msg = f"Error saving comparison report: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg) from e
