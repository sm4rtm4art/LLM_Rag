"""Interfaces for the document comparison module components."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .domain_models import (
    AlignmentPair,
    # ComparisonResultType, # Not directly used in interface signatures, but SectionComparison uses it
    DocumentFormat,
    Section,
    SectionComparison,
)

# Attempt to import type definitions from their current locations.
# These might need to be moved to a common types.py if they cause circular dependencies
# or if they are broadly used across interfaces and components.


class IParser(ABC):
    """Interface for document parsing components."""

    @abstractmethod
    def parse(self, document_content: Union[str, Path], format_type: DocumentFormat) -> List[Section]:
        """Parse document content into a list of sections.

        Args:
            document_content: The content of the document as a string or a path to the document file.
            format_type: The format of the document.

        Returns:
            A list of Section objects representing the parsed document.

        """
        pass


class IAligner(ABC):
    """Interface for section alignment components."""

    @abstractmethod
    def align_sections(
        self, source_sections: List[Section], target_sections: List[Section]
    ) -> List[Tuple[Optional[Section], Optional[Section]]]:
        """Align sections between a source and a target document.

        Args:
            source_sections: A list of Section objects from the source document.
            target_sections: A list of Section objects from the target document.

        Returns:
            A list of tuples, where each tuple represents an aligned pair of sections
            (source_section, target_section). If a section has no corresponding
            alignment, the respective element in the tuple will be None.

        """
        pass


class IComparisonEngine(ABC):
    """Interface for section comparison engine components."""

    @abstractmethod
    async def compare_sections(self, aligned_pairs: List[AlignmentPair]) -> List[SectionComparison]:
        """Compare aligned pairs of sections.

        Args:
            aligned_pairs: A list of AlignmentPair objects, where each object contains an aligned
                           pair of (source_section, target_section) and alignment metadata.

        Returns:
            A list of SectionComparison objects, each detailing the comparison
            result for a pair of sections.

        """
        pass


class IDiffFormatter(ABC):
    """Interface for diff formatting components."""

    @abstractmethod
    def format_comparisons(self, comparisons: List[SectionComparison], title: Optional[str] = None) -> str:
        """Format the comparison results into a human-readable string.

        Args:
            comparisons: A list of SectionComparison objects.
            title: An optional title for the formatted report.

        Returns:
            A string representing the formatted diff report.

        """
        pass
