"""Document comparison module for identifying differences between documents."""

# Import from component_protocols and domain_models first
# Then import implementations
from .alignment import SectionAligner
from .comparison_engine import EmbeddingComparisonEngine
from .component_protocols import IAligner, IComparisonEngine, IDiffFormatter, IParser  # If they are meant to be public
from .diff_formatter import DiffFormatter
from .document_parser import DocumentParser
from .domain_models import (
    AlignmentConfig,
    AlignmentPair,
    AlignmentStrategy,  # Added
    AnnotationStyle,
    ComparisonConfig,
    ComparisonPipelineConfig,
    ComparisonResultType,  # Changed from ComparisonResult, imported from domain_models
    DiffFormat,
    DocumentFormat,
    FormatterConfig,
    ParserConfig,  # Added
    Section,
    SectionComparison,
    SectionType,  # Added
    SimilarityThresholds,  # Added
)
from .pipeline import ComparisonPipeline

# Update __all__ accordingly
__all__ = [
    # Protocols (Interfaces)
    'IParser',
    'IAligner',
    'IComparisonEngine',
    'IDiffFormatter',
    # Domain Models & Enums
    'Section',
    'DocumentFormat',
    'SectionType',
    'ParserConfig',
    'AlignmentPair',
    'AlignmentStrategy',
    'AlignmentConfig',
    'ComparisonResultType',  # Changed from ComparisonResult
    'SimilarityThresholds',
    'ComparisonConfig',
    'SectionComparison',
    'DiffFormat',
    'AnnotationStyle',
    'FormatterConfig',
    'ComparisonPipelineConfig',
    # Implementations
    'DocumentParser',
    'SectionAligner',
    'EmbeddingComparisonEngine',
    'DiffFormatter',
    'ComparisonPipeline',
]
