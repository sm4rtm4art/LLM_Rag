#!/usr/bin/env python3
"""Example script for evaluating OCR quality.

This script demonstrates how to use the OCR evaluation module to assess OCR quality
with and without preprocessing, using sample documents from the test_subset.
"""

import argparse
import time
from pathlib import Path

from llm_rag.document_processing.ocr.evaluation import calculate_metrics, compare_preprocessing_methods
from llm_rag.document_processing.ocr.pipeline import OCRPipeline, OCRPipelineConfig


def run_evaluation(
    pdf_path: Path,
    ground_truth_path: Path = None,
    output_dir: Path = None,
    language: str = 'deu+eng',
):
    """Run OCR evaluation with and without preprocessing.

    Args:
        pdf_path: Path to the PDF file to process.
        ground_truth_path: Path to ground truth text file (optional).
        output_dir: Directory to save output (optional).
        language: Tesseract language setting (default: deu+eng for German + English).

    Returns:
        Dictionary with evaluation results.

    """
    print(f'\n[OCR Evaluation] Processing PDF: {pdf_path}')
    print(f'[OCR Evaluation] Using language: {language}')

    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create OCR pipelines with and without preprocessing
    basic_config = OCRPipelineConfig(
        # Basic settings with no preprocessing
        pdf_dpi=300,
        preprocessing_enabled=False,
        output_format='raw',  # Raw text for accurate evaluation
        languages=language,  # Set to the provided language
    )

    enhanced_config = OCRPipelineConfig(
        # Enhanced settings with preprocessing
        pdf_dpi=400,
        preprocessing_enabled=True,
        deskew_enabled=True,
        threshold_enabled=True,
        threshold_method='adaptive',
        contrast_adjust=1.2,
        sharpen_enabled=True,
        output_format='raw',  # Raw text for accurate evaluation
        languages=language,  # Set to the provided language
    )

    basic_pipeline = OCRPipeline(basic_config)
    enhanced_pipeline = OCRPipeline(enhanced_config)

    # 2. Process the PDF with both pipelines
    print('[OCR Evaluation] Running basic OCR...')
    start_time = time.time()
    basic_text = basic_pipeline.process_pdf(pdf_path)
    basic_time = time.time() - start_time
    print(f'[OCR Evaluation] Basic OCR completed in {basic_time:.2f} seconds')

    print('[OCR Evaluation] Running enhanced OCR with preprocessing...')
    start_time = time.time()
    enhanced_text = enhanced_pipeline.process_pdf(pdf_path)
    enhanced_time = time.time() - start_time
    print(f'[OCR Evaluation] Enhanced OCR completed in {enhanced_time:.2f} seconds')

    # Check for German characters in the results
    has_german_chars_basic = any(c in basic_text for c in 'äöüßÄÖÜ')
    has_german_chars_enhanced = any(c in enhanced_text for c in 'äöüßÄÖÜ')

    print(f'[OCR Evaluation] Basic OCR contains German characters: {has_german_chars_basic}')
    print(f'[OCR Evaluation] Enhanced OCR contains German characters: {has_german_chars_enhanced}')

    # 3. Save extracted text if output directory is specified
    if output_dir:
        with open(output_dir / f'{pdf_path.stem}_basic.txt', 'w', encoding='utf-8') as f:
            f.write(basic_text)
        with open(output_dir / f'{pdf_path.stem}_enhanced.txt', 'w', encoding='utf-8') as f:
            f.write(enhanced_text)

    # 4. If ground truth file exists, calculate evaluation metrics
    if ground_truth_path and Path(ground_truth_path).exists():
        print(f'[OCR Evaluation] Using ground truth text from {ground_truth_path}')
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read()

        # Compare basic and enhanced OCR against ground truth
        comparison = compare_preprocessing_methods(ground_truth, basic_text, enhanced_text, normalize=True)

        # Print results
        print('\n[OCR Evaluation] Results:')
        print(f'  Character Error Rate (Basic): {comparison["base"]["character_error_rate"]:.4f}')
        print(f'  Character Error Rate (Enhanced): {comparison["enhanced"]["character_error_rate"]:.4f}')
        print(f'  CER Improvement: {comparison["improvement_percent"]["character_error_rate"]:.2f}%')
        print(f'  Word Error Rate (Basic): {comparison["base"]["word_error_rate"]:.4f}')
        print(f'  Word Error Rate (Enhanced): {comparison["enhanced"]["word_error_rate"]:.4f}')
        print(f'  WER Improvement: {comparison["improvement_percent"]["word_error_rate"]:.2f}%')

        # Save results to output directory
        if output_dir:
            result_file = output_dir / f'{pdf_path.stem}_evaluation_results.txt'
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write('OCR Evaluation Results\n')
                f.write('=====================\n\n')
                f.write(f'PDF: {pdf_path}\n')
                f.write(f'Ground Truth: {ground_truth_path}\n')
                f.write(f'Language: {language}\n\n')
                f.write('Basic OCR:\n')
                f.write(f'  Character Error Rate: {comparison["base"]["character_error_rate"]:.4f}\n')
                f.write(f'  Word Error Rate: {comparison["base"]["word_error_rate"]:.4f}\n')
                f.write(f'  Character Accuracy: {comparison["base"]["character_accuracy"]:.4f}\n')
                f.write(f'  Word Accuracy: {comparison["base"]["word_accuracy"]:.4f}\n')
                f.write(f'  Contains German characters: {has_german_chars_basic}\n\n')
                f.write('Enhanced OCR (with preprocessing):\n')
                f.write(f'  Character Error Rate: {comparison["enhanced"]["character_error_rate"]:.4f}\n')
                f.write(f'  Word Error Rate: {comparison["enhanced"]["word_error_rate"]:.4f}\n')
                f.write(f'  Character Accuracy: {comparison["enhanced"]["character_accuracy"]:.4f}\n')
                f.write(f'  Word Accuracy: {comparison["enhanced"]["word_accuracy"]:.4f}\n')
                f.write(f'  Contains German characters: {has_german_chars_enhanced}\n\n')
                f.write('Improvement:\n')
                f.write(f'  CER Improvement: {comparison["improvement_percent"]["character_error_rate"]:.2f}%\n')
                f.write(f'  WER Improvement: {comparison["improvement_percent"]["word_error_rate"]:.2f}%\n')
                f.write(
                    f'  Character Accuracy Improvement: '
                    f'{comparison["improvement_percent"]["character_accuracy"]:.2f}%\n'
                )
                f.write(f'  Word Accuracy Improvement: {comparison["improvement_percent"]["word_accuracy"]:.2f}%\n')

            print(f'[OCR Evaluation] Results saved to {result_file}')

        return comparison
    else:
        # If no ground truth, just compare basic and enhanced text
        print('\n[OCR Evaluation] No ground truth provided. Showing basic text stats:')
        print(f'  Basic OCR text length: {len(basic_text)} characters')
        print(f'  Enhanced OCR text length: {len(enhanced_text)} characters')
        print(f'  Difference: {abs(len(enhanced_text) - len(basic_text))} characters')

        # Calculate simple metrics without ground truth
        metrics = calculate_metrics(enhanced_text, basic_text)
        print(f'  Character difference rate: {metrics["character_error_rate"]:.4f}')
        print(f'  Word difference rate: {metrics["word_error_rate"]:.4f}')

        return {
            'basic_text_length': len(basic_text),
            'enhanced_text_length': len(enhanced_text),
            'has_german_chars_basic': has_german_chars_basic,
            'has_german_chars_enhanced': has_german_chars_enhanced,
        }


def main():
    """Run the OCR evaluation example."""
    parser = argparse.ArgumentParser(description='OCR Evaluation Example')
    parser.add_argument(
        '--pdf',
        '-p',
        type=str,
        help='Path to PDF file to process',
        default='data/documents/test_subset/VDE_0636-3_A2_E__DIN_VDE_0636-3_A2__2010-04.pdf',
    )
    parser.add_argument(
        '--ground-truth',
        '-g',
        type=str,
        help='Path to ground truth text file',
        default=None,
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output directory',
        default='output/ocr_evaluation',
    )
    parser.add_argument(
        '--language',
        '-l',
        type=str,
        help="Tesseract language setting (e.g., 'deu', 'eng', 'deu+eng')",
        default='deu+eng',  # Default to German + English
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    ground_truth_path = Path(args.ground_truth) if args.ground_truth else None
    output_dir = Path(args.output)

    if not pdf_path.exists():
        print(f'Error: PDF file not found: {pdf_path}')
        return 1

    run_evaluation(
        pdf_path=pdf_path,
        ground_truth_path=ground_truth_path,
        output_dir=output_dir,
        language=args.language,
    )

    return 0


if __name__ == '__main__':
    exit(main())
