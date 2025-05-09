"""End-to-end test script for OCR followed by Document Comparison.

This script demonstrates a complete workflow:
1. Takes two PDF documents as input.
2. Processes each PDF through the OCRPipeline to extract structured Markdown.
3. Feeds the Markdown outputs into the ComparisonPipeline.
4. Prints the generated diff report.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path to allow direct execution of script from anywhere
# and ensure relative imports within the llm_rag package work correctly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from src.llm_rag.document_processing.comparison.domain_models import (
        ComparisonPipelineConfig,
        DocumentFormat,
        LLMComparerPipelineConfig,
    )
    from src.llm_rag.document_processing.comparison.pipeline import ComparisonPipeline
    from src.llm_rag.document_processing.ocr.pipeline import (
        OCRPipeline,
        OCRPipelineConfig,
    )
    from src.llm_rag.utils.errors import DocumentProcessingError
except ImportError as e:
    print(
        f'Error: Could not import necessary modules. Ensure the script is run from the project root '
        f'or the llm_rag package is correctly installed. Details: {e}'
    )
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def run_ocr_and_comparison(pdf_path1: Path, pdf_path2: Path, enable_llm: bool, llm_model: str) -> None:
    """Run the OCR pipeline on two PDFs and then compare their Markdown outputs.

    Args:
        pdf_path1: Path to the first PDF document.
        pdf_path2: Path to the second PDF document.
        enable_llm: Whether to enable LLM analysis in the comparison pipeline.
        llm_model: The LLM model name to use if LLM analysis is enabled.

    """
    logger.info(f'Starting OCR and comparison for:\n  PDF 1: {pdf_path1}\n  PDF 2: {pdf_path2}')
    logger.info(f'LLM Analysis for comparison: {"Enabled" if enable_llm else "Disabled"}')
    if enable_llm:
        logger.info(f'LLM Model for comparison: {llm_model}')

    # --- 1. Configure and Initialize OCR Pipeline ---
    # Using default OCR config, but ensuring Markdown output and enabling LLM cleaning
    ocr_config = OCRPipelineConfig(output_format='markdown', llm_cleaning_enabled=True)
    ocr_pipeline = OCRPipeline(config=ocr_config)
    logger.info('OCRPipeline initialized with LLM cleaning enabled.')

    # --- 2. Process PDFs with OCR ---
    markdown_content1_str: Optional[str] = None
    markdown_content2_str: Optional[str] = None

    try:
        logger.info(f'Processing PDF 1: {pdf_path1.name}...')
        markdown_content1_str = ocr_pipeline.process_pdf(pdf_path1)
        logger.info(f'PDF 1 processed. Markdown length: {len(markdown_content1_str)}')

        logger.info(f'Processing PDF 2: {pdf_path2.name}...')
        markdown_content2_str = ocr_pipeline.process_pdf(pdf_path2)
        logger.info(f'PDF 2 processed. Markdown length: {len(markdown_content2_str)}')

    except DocumentProcessingError as e:
        logger.error(f'Error during OCR processing: {e}', exc_info=True)
        return
    except Exception as e:
        logger.error(f'An unexpected error occurred during OCR: {e}', exc_info=True)
        return

    if not markdown_content1_str or not markdown_content2_str:
        logger.error('Failed to get Markdown content from one or both PDFs. Aborting comparison.')
        return

    # --- 3. Configure and Initialize Comparison Pipeline ---
    llm_comparer_config = LLMComparerPipelineConfig(enable_llm_analysis=enable_llm, llm_model_name=llm_model)
    comparison_pipeline_config = ComparisonPipelineConfig(llm_comparer_pipeline_config=llm_comparer_config)
    comparison_pipeline = ComparisonPipeline(config=comparison_pipeline_config)
    logger.info('ComparisonPipeline initialized.')

    # --- 4. Compare the OCR'd documents ---
    try:
        logger.info('Comparing extracted Markdown documents...')
        diff_report_str = await comparison_pipeline.compare_documents(
            source_document=markdown_content1_str,
            target_document=markdown_content2_str,
            source_format=DocumentFormat.MARKDOWN,
            target_format=DocumentFormat.MARKDOWN,
            title=f'Comparison between {pdf_path1.name} and {pdf_path2.name}',
        )
        logger.info('Document comparison completed.')

        print('\n' + '=' * 30 + ' DIFF REPORT ' + '=' * 30)
        print(diff_report_str)
        print('=' * (60 + len(' DIFF REPORT ')) + '\n')

    except DocumentProcessingError as e:
        logger.error(f'Error during document comparison: {e}', exc_info=True)
    except Exception as e:
        logger.error(f'An unexpected error occurred during comparison: {e}', exc_info=True)


def main():
    """Parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(description='End-to-end OCR and Document Comparison Test Script.')
    parser.add_argument(
        '--pdf1',
        type=str,
        default='data/documents/test_subset/VDE_0712-30_A102_E__DIN_IEC_61347-1_A102__2006-01.pdf',
        help='Path to the first PDF document (relative to project root).',
    )
    parser.add_argument(
        '--pdf2',
        type=str,
        default='data/documents/test_subset/VDE_0712-30_A103_E__DIN_IEC_61347-1_A103__2006-01.pdf',
        help='Path to the second PDF document (relative to project root).',
    )
    parser.add_argument('--enable-llm', action='store_true', help='Enable LLM analysis for document comparison.')
    parser.add_argument(
        '--llm-model',
        type=str,
        default='phi3:mini',
        help='LLM model name to use for comparison (e.g., phi3:mini, gemma:2b).',
    )

    args = parser.parse_args()

    # Construct absolute paths from project root
    pdf_path1_abs = PROJECT_ROOT / args.pdf1
    pdf_path2_abs = PROJECT_ROOT / args.pdf2

    if not pdf_path1_abs.is_file():
        logger.error(f'Error: PDF file not found at {pdf_path1_abs}')
        print(f'Please ensure PDF1 exists at: {args.pdf1} (relative to project root {PROJECT_ROOT})')
        sys.exit(1)
    if not pdf_path2_abs.is_file():
        logger.error(f'Error: PDF file not found at {pdf_path2_abs}')
        print(f'Please ensure PDF2 exists at: {args.pdf2} (relative to project root {PROJECT_ROOT})')
        sys.exit(1)

    asyncio.run(run_ocr_and_comparison(pdf_path1_abs, pdf_path2_abs, args.enable_llm, args.llm_model))


if __name__ == '__main__':
    main()
