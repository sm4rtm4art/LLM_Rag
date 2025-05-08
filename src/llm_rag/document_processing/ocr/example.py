#!/usr/bin/env python3
"""Example script demonstrating the OCR pipeline capabilities.

This script shows how to use the OCR pipeline to extract text from PDF documents,
apply image preprocessing to improve OCR quality, and save the results as
Markdown files.
"""

import argparse
import time
from pathlib import Path

from llm_rag.document_processing.ocr.pipeline import OCRPipeline, OCRPipelineConfig


def process_basic(pdf_path, output_dir=None, language='deu+eng'):
    """Process a PDF with basic OCR settings.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save output (optional).
        language: Tesseract language setting (default: deu+eng for German + English).

    Returns:
        Path to the saved Markdown file.

    """
    print(f'\n[Basic OCR] Processing: {pdf_path}')
    print(f'[Basic OCR] Using language: {language}')
    start_time = time.time()

    # Create OCR pipeline with default settings but specified language
    config = OCRPipelineConfig(languages=language, output_format='markdown')
    pipeline = OCRPipeline(config)

    # Process and save as markdown
    output_file = pipeline.save_to_markdown(pdf_path, output_dir)

    processing_time = time.time() - start_time
    print(f'[Basic OCR] Completed in {processing_time:.2f} seconds')
    print(f'[Basic OCR] Output saved to: {output_file}')

    # Check for German characters in the output
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        has_german_chars = any(c in content for c in 'äöüßÄÖÜ')
        print(f'[Basic OCR] Output contains German characters: {has_german_chars}')

    return output_file


def process_enhanced(pdf_path, output_dir=None, language='deu+eng'):
    """Process a PDF with enhanced OCR settings and preprocessing.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save output (optional).
        language: Tesseract language setting (default: deu+eng for German + English).

    Returns:
        Path to the saved Markdown file.

    """
    print(f'\n[Enhanced OCR] Processing: {pdf_path}')
    print(f'[Enhanced OCR] Using language: {language}')
    start_time = time.time()

    # Create OCR pipeline with advanced settings
    config = OCRPipelineConfig(
        # PDF settings
        pdf_dpi=400,  # Higher DPI for better quality
        # Preprocessing settings
        preprocessing_enabled=True,
        deskew_enabled=True,
        threshold_enabled=True,
        threshold_method='adaptive',
        contrast_adjust=1.2,
        sharpen_enabled=True,
        # OCR settings
        languages=language,  # German and English languages
        psm=3,  # Fully automatic page segmentation
        # Output settings
        output_format='markdown',
        detect_headings=True,
        detect_lists=True,
    )

    pipeline = OCRPipeline(config)

    # Process and save as markdown
    output_file = pipeline.save_to_markdown(pdf_path, output_dir)

    processing_time = time.time() - start_time
    print(f'[Enhanced OCR] Completed in {processing_time:.2f} seconds')
    print(f'[Enhanced OCR] Output saved to: {output_file}')

    # Check for German characters in the output
    with open(output_file, 'r', encoding='utf-8') as f:
        content = f.read()
        has_german_chars = any(c in content for c in 'äöüßÄÖÜ')
        print(f'[Enhanced OCR] Output contains German characters: {has_german_chars}')

    return output_file


def main():
    """Run the OCR pipeline example."""
    parser = argparse.ArgumentParser(description='OCR Pipeline Example')
    parser.add_argument(
        '--pdf',
        '-p',
        type=str,
        help='Path to PDF file to process',
        default='data/documents/test_subset/VDE_0636-3_A2_E__DIN_VDE_0636-3_A2__2010-04.pdf',
    )
    parser.add_argument('--output', '-o', type=str, help='Output directory', default='output/ocr')
    parser.add_argument(
        '--mode', '-m', type=str, choices=['basic', 'enhanced', 'both'], default='both', help='OCR processing mode'
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
    output_dir = Path(args.output)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        print(f'Error: PDF file not found: {pdf_path}')
        return 1

    # Process according to selected mode
    if args.mode in ['basic', 'both']:
        basic_output = process_basic(pdf_path, output_dir / 'basic', args.language)

    if args.mode in ['enhanced', 'both']:
        enhanced_output = process_enhanced(pdf_path, output_dir / 'enhanced', args.language)

    if args.mode == 'both':
        print('\nComparison complete!')
        print(f'Basic output: {basic_output}')
        print(f'Enhanced output: {enhanced_output}')
        print('\nYou can now compare the OCR results to see the improvements from preprocessing.')

    return 0


if __name__ == '__main__':
    exit(main())
