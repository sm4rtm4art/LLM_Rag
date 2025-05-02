"""Document comparison module for identifying differences between documents."""

from llm_rag.document_processing.comparison.alignment import AlignmentConfig, AlignmentPair, SectionAligner
from llm_rag.document_processing.comparison.comparison_engine import (
    ComparisonConfig,
    ComparisonResult,
    EmbeddingComparisonEngine,
    SectionComparison,
)
from llm_rag.document_processing.comparison.diff_formatter import (
    AnnotationStyle,
    DiffFormat,
    DiffFormatter,
    FormatterConfig,
)
from llm_rag.document_processing.comparison.document_parser import DocumentFormat, DocumentParser, Section
from llm_rag.document_processing.comparison.pipeline import ComparisonPipeline, ComparisonPipelineConfig

__all__ = [
    # Document parsing
    "DocumentParser",
    "Section",
    "DocumentFormat",
    # Section alignment
    "SectionAligner",
    "AlignmentPair",
    "AlignmentConfig",
    # Comparison engine
    "EmbeddingComparisonEngine",
    "ComparisonConfig",
    "SectionComparison",
    "ComparisonResult",
    # Diff formatting
    "DiffFormatter",
    "FormatterConfig",
    "DiffFormat",
    "AnnotationStyle",
    # Pipeline
    "ComparisonPipeline",
    "ComparisonPipelineConfig",
]
