"""Type definitions for the document comparison module."""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional  # Pattern for re.compile

# --- Common Base Types (if any, or Pydantic models if used project-wide) ---

# --- Types for document_parser.py ---


class DocumentFormat(Enum):
    """Enum for supported document formats."""

    MARKDOWN = 'markdown'
    JSON = 'json'
    TEXT = 'text'
    # Add other formats as needed, e.g., PDF_TEXT_LAYER


class SectionType(Enum):
    """Types of document sections."""

    HEADING = 'heading'
    PARAGRAPH = 'paragraph'
    LIST = 'list'
    TABLE = 'table'
    CODE = 'code'
    IMAGE = 'image'
    UNKNOWN = 'unknown'


@dataclass
class Section:
    """Represents a section of a document."""

    title: str
    content: str
    level: int
    section_type: Optional[SectionType] = None  # Added section_type
    parent: Optional['Section'] = None
    children: List['Section'] = field(default_factory=list)
    section_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Optional[dict] = field(default_factory=dict)  # Added for extra info like language for code block
    # Add other relevant attributes like page_number, coordinates if available from parsing

    def __post_init__(self):
        """Ensure section_id is unique if not provided after initialization."""
        if not self.section_id:
            self.section_id = str(uuid.uuid4())

    def __hash__(self) -> int:
        """Return a hash based on the section_id."""
        return hash(self.section_id)

    def __eq__(self, other: Any) -> bool:
        """Check equality based on the section_id."""
        if not isinstance(other, Section):
            return NotImplemented
        return self.section_id == other.section_id


@dataclass
class ParserConfig:
    """Configuration for the DocumentParser."""

    default_format: DocumentFormat = DocumentFormat.MARKDOWN
    min_section_length: int = 10
    split_on_headings: bool = True
    split_on_blank_lines: bool = True
    heading_patterns: List[str] = field(
        default_factory=lambda: [
            r'^#{1,6}\s+(.+)$',  # ATX style headings
            r'^(.+)\n[=]{2,}$',  # Setext style H1
            r'^(.+)\n[-]{2,}$',  # Setext style H2
        ]
    )
    # max_chunk_size: Optional[int] = None # This was from a simpler version,
    # can be added if fixed-chunking is a primary strategy


# --- Types for alignment.py ---


class AlignmentStrategy(Enum):
    """Enum for different section alignment strategies."""

    HEADING_MATCH = 'heading_match'
    SEQUENCE_ALIGNMENT = 'sequence_alignment'
    CONTENT_SIMILARITY = 'content_similarity'
    HYBRID = 'hybrid'  # Added HYBRID based on alignment.py usage


@dataclass
class AlignmentConfig:
    """Configuration for the SectionAligner."""

    strategy: AlignmentStrategy = AlignmentStrategy.HYBRID  # Defaulting to HYBRID
    similarity_threshold: float = 0.7  # Kept from alignment.py
    use_sequence_information: bool = True  # From alignment.py AlignmentConfig
    heading_weight: float = 2.0  # From alignment.py AlignmentConfig
    content_weight: float = 1.0  # From alignment.py AlignmentConfig
    max_gap_penalty: int = 3  # From alignment.py AlignmentConfig


@dataclass
class AlignmentPair:
    """Represents a pair of aligned sections from two documents."""

    source_section: Optional[Section]
    target_section: Optional[Section]
    similarity_score: float = 0.0
    method: Optional[AlignmentStrategy] = (
        None  # Changed from AlignmentMethod (local in alignment.py) to AlignmentStrategy
    )

    @property
    def is_source_only(self) -> bool:
        """Check if this alignment represents a source-only section (e.g., a deletion)."""
        return self.source_section is not None and self.target_section is None

    @property
    def is_target_only(self) -> bool:
        """Check if this alignment represents a target-only section (e.g., an addition)."""
        return self.source_section is None and self.target_section is not None

    @property
    def is_aligned(self) -> bool:
        """Check if this alignment represents a valid pairing between a source and a target section."""
        return self.source_section is not None and self.target_section is not None


# --- Types for comparison_engine.py ---


class ComparisonResultType(Enum):
    """Enum for the type of comparison result between two sections."""

    SIMILAR = 'similar'
    DIFFERENT = 'different'
    MODIFIED = 'modified'
    NEW = 'new'
    DELETED = 'deleted'
    # Note: comparison_engine.py also had MINOR_CHANGES, MAJOR_CHANGES, REWRITTEN.
    # These need to be mapped to the above, or this Enum needs to be expanded.
    # For now, MODIFIED and DIFFERENT cover these.


@dataclass
class SimilarityThresholds:
    """Thresholds for classifying similarity."""

    similar: float = 0.95
    modified: float = 0.8  # For MODIFIED type
    different: float = 0.6  # For DIFFERENT type
    # Removed major_change_threshold, rewritten_threshold from comparison_engine's old config


@dataclass
class ComparisonConfig:
    """Configuration for the EmbeddingComparisonEngine."""

    embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    similarity_thresholds: SimilarityThresholds = field(default_factory=SimilarityThresholds)
    # chunk_size: int = 512 # From comparison_engine's old config, can be added if needed


@dataclass
class SectionComparison:
    """Represents the result of comparing two sections."""

    result_type: ComparisonResultType
    # alignment_pair: AlignmentPair # This was in comparison_engine.py's SectionComparison. Good to have.
    source_section: Optional[Section] = None  # Redundant if alignment_pair is used
    target_section: Optional[Section] = None  # Redundant if alignment_pair is used
    similarity_score: Optional[float] = None
    details: Optional[str] = None
    # Adding alignment_pair here for completeness
    alignment_pair: Optional[AlignmentPair] = None


# --- Types for diff_formatter.py ---


class DiffFormat(Enum):
    """Enum for different output formats of the diff report."""

    MARKDOWN = 'markdown'
    HTML = 'html'
    TEXT = 'text'


class AnnotationStyle(Enum):
    """Styles for annotating differences."""  # Moved from diff_formatter.py

    STANDARD = 'standard'
    DETAILED = 'detailed'
    MINIMAL = 'minimal'


@dataclass
class FormatterConfig:
    """Configuration for the DiffFormatter."""

    output_format: DiffFormat = DiffFormat.MARKDOWN
    annotation_style: AnnotationStyle = AnnotationStyle.STANDARD  # Added
    show_similar_content: bool = False  # Renamed from show_unchanged for clarity
    detail_level: int = 1  # Kept from old FormatterConfig in types.py
    include_metadata: bool = False  # From diff_formatter.py FormatterConfig
    include_similarity_scores: bool = True  # From diff_formatter.py FormatterConfig
    wrap_width: int = 100  # From diff_formatter.py FormatterConfig
    color_output: bool = False  # From diff_formatter.py FormatterConfig


# --- Types for pipeline.py ---


@dataclass
class ComparisonPipelineConfig:
    """Configuration for the document comparison pipeline."""

    parser_config: ParserConfig = field(default_factory=ParserConfig)
    alignment_config: AlignmentConfig = field(default_factory=AlignmentConfig)
    comparison_config: ComparisonConfig = field(default_factory=ComparisonConfig)
    formatter_config: FormatterConfig = field(default_factory=FormatterConfig)
    cache_intermediate_results: bool = False
    # Add other pipeline-level configurations
