"""Evaluation utilities for OCR output quality.

This module provides tools to evaluate the quality of OCR output by comparing
against ground truth text, calculating metrics like Character Error Rate (CER)
and Word Error Rate (WER).
"""

import difflib
import re
from typing import Dict, List, Union

from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_character_error_rate(ground_truth: str, ocr_text: str, normalize: bool = True) -> float:
    """Calculate Character Error Rate between ground truth and OCR output.

    Character Error Rate is defined as the Levenshtein distance between the
    ground truth and OCR text, divided by the length of the ground truth text.

    Args:
        ground_truth: The reference or ground truth text.
        ocr_text: The OCR-generated text to evaluate.
        normalize: Whether to normalize the text before comparison (remove
            whitespace, convert to lowercase).

    Returns:
        The character error rate as a float between 0 and 1.

    """
    if not ground_truth:
        logger.warning("Ground truth is empty, returning error rate of 1.0")
        return 1.0

    if normalize:
        # Remove all whitespace and convert to lowercase to focus on character accuracy
        truth_chars = "".join(re.sub(r"\s+", "", ground_truth).lower())
        ocr_chars = "".join(re.sub(r"\s+", "", ocr_text).lower())
    else:
        truth_chars = ground_truth
        ocr_chars = ocr_text

    # Use difflib's SequenceMatcher to compute the similarity
    # This is an efficient way to calculate edit distance
    matcher = difflib.SequenceMatcher(None, truth_chars, ocr_chars)
    similarity = matcher.ratio()

    # CER = 1 - similarity
    return 1.0 - similarity


def calculate_word_error_rate(ground_truth: str, ocr_text: str, normalize: bool = True) -> float:
    """Calculate Word Error Rate between ground truth and OCR output.

    Word Error Rate is defined as the edit distance between the word
    sequences of the ground truth and OCR text, divided by the number
    of words in the ground truth.

    Args:
        ground_truth: The reference or ground truth text.
        ocr_text: The OCR-generated text to evaluate.
        normalize: Whether to normalize the text before comparison (lowercase,
            remove punctuation).

    Returns:
        The word error rate as a float between 0 and 1.

    """
    if not ground_truth:
        logger.warning("Ground truth is empty, returning error rate of 1.0")
        return 1.0

    if normalize:
        # Convert to lowercase and remove punctuation
        ground_truth = re.sub(r"[^\w\s]", "", ground_truth.lower())
        ocr_text = re.sub(r"[^\w\s]", "", ocr_text.lower())

    # Split into words
    truth_words = ground_truth.split()
    ocr_words = ocr_text.split()

    if not truth_words:
        logger.warning("No words in ground truth after normalization, returning error rate of 1.0")
        return 1.0

    # Use difflib's SequenceMatcher to compute the similarity
    matcher = difflib.SequenceMatcher(None, truth_words, ocr_words)
    similarity = matcher.ratio()

    # WER = 1 - similarity
    return 1.0 - similarity


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings.

    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change
    one string into the other.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The Levenshtein distance between the two strings.

    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_metrics(ground_truth: str, ocr_text: str, normalize: bool = True) -> Dict[str, float]:
    """Calculate various OCR quality metrics.

    Computes multiple metrics for evaluating OCR quality:
    - Character Error Rate (CER)
    - Word Error Rate (WER)

    Args:
        ground_truth: The reference or ground truth text.
        ocr_text: The OCR-generated text to evaluate.
        normalize: Whether to normalize the text before comparison.

    Returns:
        Dictionary containing the computed metrics.

    """
    cer = calculate_character_error_rate(ground_truth, ocr_text, normalize)
    wer = calculate_word_error_rate(ground_truth, ocr_text, normalize)

    return {
        "character_error_rate": cer,
        "word_error_rate": wer,
        "character_accuracy": 1.0 - cer,
        "word_accuracy": 1.0 - wer,
    }


def evaluate_document(
    ground_truth: Union[str, List[str]], ocr_result: Union[str, List[str]], normalize: bool = True
) -> Dict[str, float]:
    """Evaluate OCR quality for a complete document.

    For multi-page documents provided as lists of strings,
    calculates the average metrics across all pages.

    Args:
        ground_truth: The reference text, either as a single string
            or as a list of strings (one per page).
        ocr_result: The OCR-generated text to evaluate, in the same
            format as ground_truth.
        normalize: Whether to normalize the text before comparison.

    Returns:
        Dictionary containing the computed metrics.

    """
    # Convert single strings to lists for uniform processing
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    if isinstance(ocr_result, str):
        ocr_result = [ocr_result]

    # Ensure both inputs have the same number of pages
    if len(ground_truth) != len(ocr_result):
        logger.warning(
            f"Number of pages in ground truth ({len(ground_truth)}) and OCR result "
            f"({len(ocr_result)}) don't match. Evaluating only the first "
            f"min({len(ground_truth)}, {len(ocr_result)}) pages."
        )
        num_pages = min(len(ground_truth), len(ocr_result))
        ground_truth = ground_truth[:num_pages]
        ocr_result = ocr_result[:num_pages]

    # Calculate metrics for each page
    all_metrics = []
    for i, (truth_page, ocr_page) in enumerate(zip(ground_truth, ocr_result, strict=False)):
        page_metrics = calculate_metrics(truth_page, ocr_page, normalize)
        all_metrics.append(page_metrics)
        logger.debug(
            f"Page {i + 1} metrics: CER={page_metrics['character_error_rate']:.4f}, "
            f"WER={page_metrics['word_error_rate']:.4f}"
        )

    # Calculate average metrics across all pages
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = sum(page[metric] for page in all_metrics) / len(all_metrics)

    return avg_metrics


def compare_preprocessing_methods(
    ground_truth: str, base_ocr: str, enhanced_ocr: str, normalize: bool = True
) -> Dict[str, Dict[str, float]]:
    """Compare OCR quality with and without preprocessing.

    Args:
        ground_truth: The reference text.
        base_ocr: OCR text without preprocessing.
        enhanced_ocr: OCR text with preprocessing.
        normalize: Whether to normalize texts before comparison.

    Returns:
        Dictionary with base and enhanced metrics and the improvement percentages.

    """
    base_metrics = calculate_metrics(ground_truth, base_ocr, normalize)
    enhanced_metrics = calculate_metrics(ground_truth, enhanced_ocr, normalize)

    # Calculate improvement percentages
    improvement = {}
    for metric in base_metrics:
        if "error" in metric:
            # For error rates, lower is better
            if base_metrics[metric] > 0:  # Avoid division by zero
                improvement[metric] = (base_metrics[metric] - enhanced_metrics[metric]) / base_metrics[metric] * 100
            else:
                improvement[metric] = -1.0
        else:
            # For accuracy metrics, higher is better
            if base_metrics[metric] > 0:  # Avoid division by zero
                improvement[metric] = (enhanced_metrics[metric] - base_metrics[metric]) / base_metrics[metric] * 100
            else:
                improvement[metric] = 0.0

    return {"base": base_metrics, "enhanced": enhanced_metrics, "improvement_percent": improvement}
