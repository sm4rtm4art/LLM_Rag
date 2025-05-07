"""Module for aligning sections between two documents for comparison."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from llm_rag.document_processing.comparison.document_parser import Section
from llm_rag.utils.errors import DocumentProcessingError
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


class AlignmentMethod(Enum):
	"""Methods for aligning document sections."""

	HEADING_MATCH = 'heading_match'
	SEQUENCE = 'sequence'
	CONTENT_SIMILARITY = 'content_similarity'
	HYBRID = 'hybrid'


@dataclass
class AlignmentPair:
	"""Represents a pair of aligned sections from two documents.

	This class holds references to corresponding sections between documents,
	along with metadata about the alignment.
	"""

	source_section: Optional[Section]
	target_section: Optional[Section]
	similarity_score: float = 0.0
	method: AlignmentMethod = AlignmentMethod.HEADING_MATCH

	@property
	def is_source_only(self) -> bool:
		"""Check if this is a source-only alignment (deletion)."""
		return self.source_section is not None and self.target_section is None

	@property
	def is_target_only(self) -> bool:
		"""Check if this is a target-only alignment (addition)."""
		return self.source_section is None and self.target_section is not None

	@property
	def is_aligned(self) -> bool:
		"""Check if this is a valid alignment between two sections."""
		return self.source_section is not None and self.target_section is not None


@dataclass
class AlignmentConfig:
	"""Configuration for section alignment.

	Attributes:
	    method: The alignment method to use.
	    similarity_threshold: Threshold for considering sections similar.
	    use_sequence_information: Whether to consider section order.
	    heading_weight: Weight for heading-based matches (higher values favor headings).
	    content_weight: Weight for content-based matches.
	    max_gap_penalty: Maximum penalty for sequence gaps.

	"""

	method: AlignmentMethod = AlignmentMethod.HYBRID
	similarity_threshold: float = 0.7
	use_sequence_information: bool = True
	heading_weight: float = 2.0
	content_weight: float = 1.0
	max_gap_penalty: int = 3


class SectionAligner:
	"""Aligns corresponding sections between two documents.

	This class implements several strategies for matching sections between
	documents, including heading matching, sequence alignment, and content
	similarity.
	"""

	def __init__(self, config: Optional[AlignmentConfig] = None):
		"""Initialize the section aligner.

		Args:
		    config: Configuration for alignment behavior. If None, defaults are used.

		"""
		self.config = config or AlignmentConfig()
		logger.info(f'Initialized SectionAligner with method: {self.config.method.value}')

	def align_sections(self, source_sections: List[Section], target_sections: List[Section]) -> List[AlignmentPair]:
		"""Align corresponding sections between two documents.

		Args:
		    source_sections: Sections from the source document.
		    target_sections: Sections from the target document.

		Returns:
		    A list of alignment pairs mapping sections between documents.

		Raises:
		    DocumentProcessingError: If alignment fails.

		"""
		try:
			logger.debug(f'Aligning {len(source_sections)} source sections with {len(target_sections)} target sections')

			# Choose alignment method based on configuration
			if self.config.method == AlignmentMethod.HEADING_MATCH:
				return self._align_by_headings(source_sections, target_sections)
			elif self.config.method == AlignmentMethod.SEQUENCE:
				return self._align_by_sequence(source_sections, target_sections)
			elif self.config.method == AlignmentMethod.CONTENT_SIMILARITY:
				return self._align_by_content_similarity(source_sections, target_sections)
			elif self.config.method == AlignmentMethod.HYBRID:
				return self._align_hybrid(source_sections, target_sections)
			else:
				raise ValueError(f'Unsupported alignment method: {self.config.method}')

		except Exception as e:
			error_msg = f'Error aligning document sections: {str(e)}'
			logger.error(error_msg)
			raise DocumentProcessingError(error_msg) from e

	def _align_by_headings(self, source_sections: List[Section], target_sections: List[Section]) -> List[AlignmentPair]:
		"""Align sections based on heading matches.

		This method aligns sections by matching headings exactly or through
		semantic similarity. Paragraphs following matched headings are
		aligned in sequence.

		Args:
		    source_sections: Sections from the source document.
		    target_sections: Sections from the target document.

		Returns:
		    A list of alignment pairs.

		"""
		logger.debug('Aligning sections by headings')
		alignment_pairs = []

		# Extract headings and create lookup tables
		source_headings = {
			i: section for i, section in enumerate(source_sections) if section.section_type.value == 'heading'
		}
		target_headings = {
			i: section for i, section in enumerate(target_sections) if section.section_type.value == 'heading'
		}

		# Match headings (exact matches first)
		heading_matches = {}

		# Map of source heading index to matching target heading index
		for source_idx, source_heading in source_headings.items():
			for target_idx, target_heading in target_headings.items():
				# Check for exact match
				if source_heading.content.strip() == target_heading.content.strip():
					heading_matches[source_idx] = target_idx
					alignment_pairs.append(
						AlignmentPair(
							source_section=source_heading,
							target_section=target_heading,
							similarity_score=1.0,
							method=AlignmentMethod.HEADING_MATCH,
						)
					)
					break

		# Create sections map for each heading in source and target
		source_heading_to_sections = self._map_heading_to_sections(source_sections)
		target_heading_to_sections = self._map_heading_to_sections(target_sections)

		# Align sections under matched headings
		for source_heading_idx, target_heading_idx in heading_matches.items():
			source_content_idx = source_heading_to_sections.get(source_heading_idx, [])
			target_content_idx = target_heading_to_sections.get(target_heading_idx, [])

			# Simple sequence alignment of content under matched headings
			self._align_content_sequence(
				source_sections, target_sections, source_content_idx, target_content_idx, alignment_pairs
			)

		# Add unmatched source sections as source-only (deleted)
		all_source_matched = {pair.source_section for pair in alignment_pairs if pair.source_section}
		for _i, section in enumerate(source_sections):
			if section not in all_source_matched:
				alignment_pairs.append(
					AlignmentPair(
						source_section=section,
						target_section=None,
						similarity_score=0.0,
						method=AlignmentMethod.HEADING_MATCH,
					)
				)

		# Add unmatched target sections as target-only (added)
		all_target_matched = {pair.target_section for pair in alignment_pairs if pair.target_section}
		for _i, section in enumerate(target_sections):
			if section not in all_target_matched:
				alignment_pairs.append(
					AlignmentPair(
						source_section=None,
						target_section=section,
						similarity_score=0.0,
						method=AlignmentMethod.HEADING_MATCH,
					)
				)

		logger.debug(f'Found {len(alignment_pairs)} alignment pairs by heading matching')
		return alignment_pairs

	def _map_heading_to_sections(self, sections: List[Section]) -> Dict[int, List[int]]:
		"""Create a mapping from heading indices to the sections under them.

		Args:
		    sections: List of all sections in a document.

		Returns:
		    Dictionary mapping heading indices to lists of content section indices.

		"""
		heading_to_sections = {}
		last_heading_idx = None

		for i, section in enumerate(sections):
			if section.section_type.value == 'heading':
				last_heading_idx = i
				heading_to_sections[i] = []
			elif last_heading_idx is not None:
				heading_to_sections[last_heading_idx].append(i)

		return heading_to_sections

	def _align_content_sequence(
		self,
		source_sections: List[Section],
		target_sections: List[Section],
		source_idx: List[int],
		target_idx: List[int],
		alignment_pairs: List[AlignmentPair],
	) -> None:
		"""Align content sections in sequence under matched headings.

		Args:
		    source_sections: All sections from source document.
		    target_sections: All sections from target document.
		    source_idx: Indices of source sections under a heading.
		    target_idx: Indices of target sections under a heading.
		    alignment_pairs: List of alignment pairs to append results to.

		"""
		# Simple sequence alignment - match by position
		for i in range(max(len(source_idx), len(target_idx))):
			if i < len(source_idx) and i < len(target_idx):
				# Matched content by sequence
				source_section = source_sections[source_idx[i]]
				target_section = target_sections[target_idx[i]]

				# Calculate basic content similarity
				similarity = self._calculate_text_similarity(source_section.content, target_section.content)

				alignment_pairs.append(
					AlignmentPair(
						source_section=source_section,
						target_section=target_section,
						similarity_score=similarity,
						method=AlignmentMethod.SEQUENCE,
					)
				)
			elif i < len(source_idx):
				# Source-only section (deleted)
				alignment_pairs.append(
					AlignmentPair(
						source_section=source_sections[source_idx[i]],
						target_section=None,
						similarity_score=0.0,
						method=AlignmentMethod.SEQUENCE,
					)
				)
			else:
				# Target-only section (added)
				alignment_pairs.append(
					AlignmentPair(
						source_section=None,
						target_section=target_sections[target_idx[i]],
						similarity_score=0.0,
						method=AlignmentMethod.SEQUENCE,
					)
				)

	def _align_by_sequence(self, source_sections: List[Section], target_sections: List[Section]) -> List[AlignmentPair]:
		"""Align sections based on sequence position and content similarity.

		This method uses a dynamic programming approach to find the optimal
		alignment between section sequences, similar to sequence alignment
		in bioinformatics.

		Args:
		    source_sections: Sections from the source document.
		    target_sections: Sections from the target document.

		Returns:
		    A list of alignment pairs.

		"""
		logger.debug('Aligning sections by sequence')

		# Initialize similarity matrix
		similarity_matrix = np.zeros((len(source_sections), len(target_sections)))

		# Calculate similarity between all section pairs
		for i, source_section in enumerate(source_sections):
			for j, target_section in enumerate(target_sections):
				similarity = self._calculate_text_similarity(source_section.content, target_section.content)
				similarity_matrix[i, j] = similarity

		# Use dynamic programming for sequence alignment
		alignment = self._sequence_alignment(similarity_matrix)

		# Convert alignment to pairs
		alignment_pairs = []
		for i, j in alignment:
			if i is not None and j is not None:
				# Matched pair
				alignment_pairs.append(
					AlignmentPair(
						source_section=source_sections[i],
						target_section=target_sections[j],
						similarity_score=similarity_matrix[i, j],
						method=AlignmentMethod.SEQUENCE,
					)
				)
			elif i is not None:
				# Source-only (deletion)
				alignment_pairs.append(
					AlignmentPair(
						source_section=source_sections[i],
						target_section=None,
						similarity_score=0.0,
						method=AlignmentMethod.SEQUENCE,
					)
				)
			elif j is not None:
				# Target-only (addition)
				alignment_pairs.append(
					AlignmentPair(
						source_section=None,
						target_section=target_sections[j],
						similarity_score=0.0,
						method=AlignmentMethod.SEQUENCE,
					)
				)

		logger.debug(f'Found {len(alignment_pairs)} alignment pairs by sequence')
		return alignment_pairs

	def _sequence_alignment(self, similarity_matrix: np.ndarray) -> List[Tuple[Optional[int], Optional[int]]]:
		"""Perform sequence alignment using dynamic programming.

		This is a simplified version of the Needleman-Wunsch algorithm.

		Args:
		    similarity_matrix: Matrix of similarity scores between sections.

		Returns:
		    List of tuples representing the aligned indices.

		"""
		if similarity_matrix.size == 0:
			return []

		m, n = similarity_matrix.shape

		# Initialize score and traceback matrices
		score = np.zeros((m + 1, n + 1))
		traceback = np.zeros((m + 1, n + 1), dtype=int)

		# Gap penalty
		gap_penalty = -0.5

		# Fill the score and traceback matrices
		for i in range(1, m + 1):
			score[i, 0] = i * gap_penalty
			traceback[i, 0] = 1  # Up

		for j in range(1, n + 1):
			score[0, j] = j * gap_penalty
			traceback[0, j] = 2  # Left

		for i in range(1, m + 1):
			for j in range(1, n + 1):
				match_score = score[i - 1, j - 1] + similarity_matrix[i - 1, j - 1]
				delete_score = score[i - 1, j] + gap_penalty
				insert_score = score[i, j - 1] + gap_penalty

				max_score = max(match_score, delete_score, insert_score)
				score[i, j] = max_score

				if max_score == match_score:
					traceback[i, j] = 0  # Diagonal
				elif max_score == delete_score:
					traceback[i, j] = 1  # Up
				else:
					traceback[i, j] = 2  # Left

		# Traceback to find alignment
		alignment = []
		i, j = m, n

		while i > 0 or j > 0:
			if i > 0 and j > 0 and traceback[i, j] == 0:
				# Diagonal - match
				alignment.append((i - 1, j - 1))
				i -= 1
				j -= 1
			elif i > 0 and traceback[i, j] == 1:
				# Up - deletion
				alignment.append((i - 1, None))
				i -= 1
			else:
				# Left - insertion
				alignment.append((None, j - 1))
				j -= 1

		# Reverse alignment to get correct order
		return list(reversed(alignment))

	def _align_by_content_similarity(
		self, source_sections: List[Section], target_sections: List[Section]
	) -> List[AlignmentPair]:
		"""Align sections based purely on content similarity.

		This method compares each source section to each target section
		and matches the most similar pairs.

		Args:
		    source_sections: Sections from the source document.
		    target_sections: Sections from the target document.

		Returns:
		    A list of alignment pairs.

		"""
		logger.debug('Aligning sections by content similarity')

		# Calculate similarity between all section pairs
		similarity_scores = []
		for i, source_section in enumerate(source_sections):
			for j, target_section in enumerate(target_sections):
				similarity = self._calculate_text_similarity(source_section.content, target_section.content)
				if similarity >= self.config.similarity_threshold:
					similarity_scores.append((i, j, similarity))

		# Sort by similarity score (highest first)
		similarity_scores.sort(key=lambda x: x[2], reverse=True)

		# Greedy matching (best matches first)
		matched_source = set()
		matched_target = set()
		alignment_pairs = []

		for source_idx, target_idx, score in similarity_scores:
			if source_idx not in matched_source and target_idx not in matched_target:
				matched_source.add(source_idx)
				matched_target.add(target_idx)

				alignment_pairs.append(
					AlignmentPair(
						source_section=source_sections[source_idx],
						target_section=target_sections[target_idx],
						similarity_score=score,
						method=AlignmentMethod.CONTENT_SIMILARITY,
					)
				)

		# Add unmatched source sections (deleted)
		for _, section in enumerate(source_sections):
			if _ not in matched_source:
				alignment_pairs.append(
					AlignmentPair(
						source_section=section,
						target_section=None,
						similarity_score=0.0,
						method=AlignmentMethod.CONTENT_SIMILARITY,
					)
				)

		# Add unmatched target sections (added)
		for _, section in enumerate(target_sections):
			if _ not in matched_target:
				alignment_pairs.append(
					AlignmentPair(
						source_section=None,
						target_section=section,
						similarity_score=0.0,
						method=AlignmentMethod.CONTENT_SIMILARITY,
					)
				)

		logger.debug(f'Found {len(alignment_pairs)} alignment pairs by content similarity')
		return alignment_pairs

	def _align_hybrid(self, source_sections: List[Section], target_sections: List[Section]) -> List[AlignmentPair]:
		"""Align sections using a hybrid approach.

		This method combines heading matching, sequence alignment, and content similarity
		to provide more accurate alignments.

		Args:
		    source_sections: Sections from the source document.
		    target_sections: Sections from the target document.

		Returns:
		    A list of alignment pairs.

		"""
		logger.debug('Aligning sections using hybrid approach')

		# First, align headings
		heading_pairs = self._align_by_headings(
			[s for s in source_sections if s.section_type.value == 'heading'],
			[s for s in target_sections if s.section_type.value == 'heading'],
		)

		# Use heading alignment to segment the documents
		source_segments = self._split_sections_by_headings(source_sections)
		target_segments = self._split_sections_by_headings(target_sections)

		# Match heading segments
		segment_matches = self._match_segments_by_headings(source_segments, target_segments, heading_pairs)

		# Align content within matched segments using sequence alignment
		alignment_pairs = []

		for source_segment_idx, target_segment_idx in segment_matches:
			source_segment = source_segments[source_segment_idx]
			target_segment = target_segments[target_segment_idx]

			# Align content sections within segments
			segment_pairs = self._align_by_sequence(source_segment, target_segment)
			alignment_pairs.extend(segment_pairs)

		# Handle unmatched segments
		matched_source_segments = {idx for idx, _ in segment_matches}
		matched_target_segments = {idx for _, idx in segment_matches}

		# Add unmatched source segments (deleted)
		for i, segment in enumerate(source_segments):
			if i not in matched_source_segments:
				for _, section in enumerate(segment):
					alignment_pairs.append(
						AlignmentPair(
							source_section=section,
							target_section=None,
							similarity_score=0.0,
							method=AlignmentMethod.HYBRID,
						)
					)

		# Add unmatched target segments (added)
		for j, segment in enumerate(target_segments):
			if j not in matched_target_segments:
				for _, section in enumerate(segment):
					alignment_pairs.append(
						AlignmentPair(
							source_section=None,
							target_section=section,
							similarity_score=0.0,
							method=AlignmentMethod.HYBRID,
						)
					)

		logger.debug(f'Found {len(alignment_pairs)} alignment pairs using hybrid approach')
		return alignment_pairs

	def _split_sections_by_headings(self, sections: List[Section]) -> List[List[Section]]:
		"""Split sections into segments based on headings.

		Args:
		    sections: List of all sections in a document.

		Returns:
		    List of section segments, where each segment starts with a heading.

		"""
		segments = []
		current_segment = []

		for section in sections:
			if section.section_type.value == 'heading' and current_segment:
				segments.append(current_segment)
				current_segment = [section]
			else:
				current_segment.append(section)

		# Add the last segment
		if current_segment:
			segments.append(current_segment)

		return segments

	def _match_segments_by_headings(
		self,
		source_segments: List[List[Section]],
		target_segments: List[List[Section]],
		heading_pairs: List[AlignmentPair],
	) -> List[Tuple[int, int]]:
		"""Match source and target segments based on heading alignment.

		Args:
		    source_segments: Segmented source sections.
		    target_segments: Segmented target sections.
		    heading_pairs: Aligned heading pairs.

		Returns:
		    List of (source_segment_idx, target_segment_idx) matches.

		"""
		segment_matches = []

		# Create lookup dictionaries for heading sections
		source_heading_to_segment = {}
		for i, segment in enumerate(source_segments):
			if segment and segment[0].section_type.value == 'heading':
				source_heading_to_segment[segment[0]] = i

		target_heading_to_segment = {}
		for j, segment in enumerate(target_segments):
			if segment and segment[0].section_type.value == 'heading':
				target_heading_to_segment[segment[0]] = j

		# Match segments based on heading alignment
		for pair in heading_pairs:
			if pair.source_section and pair.target_section:
				if (
					pair.source_section in source_heading_to_segment
					and pair.target_section in target_heading_to_segment
				):
					source_idx = source_heading_to_segment[pair.source_section]
					target_idx = target_heading_to_segment[pair.target_section]
					segment_matches.append((source_idx, target_idx))

		return segment_matches

	def _calculate_text_similarity(self, text1: str, text2: str) -> float:
		"""Calculate basic text similarity between two strings.

		This is a simple similarity measure based on token overlap.
		In a real implementation, this would use embeddings or more
		sophisticated text similarity algorithms.

		Args:
		    text1: First text string.
		    text2: Second text string.

		Returns:
		    Similarity score between 0 and 1.

		"""
		# Simple word overlap similarity
		words1 = set(text1.lower().split())
		words2 = set(text2.lower().split())

		if not words1 or not words2:
			return 0.0

		intersection = words1.intersection(words2)
		union = words1.union(words2)

		jaccard = len(intersection) / len(union)
		return jaccard
