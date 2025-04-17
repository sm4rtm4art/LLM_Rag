"""Examples of using the optimized OCR pipeline.

This module demonstrates how to use the optimized OCR pipeline for improved performance
through caching, parallel processing, and batch processing.
"""

import argparse
import glob
import os
import time
from pathlib import Path
from typing import List, Optional

from llm_rag.document_processing.ocr.optimized_pipeline import OptimizedOCRConfig, OptimizedOCRPipeline
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


def process_single_document(
    pdf_path: str,
    use_parallel: bool = True,
    use_cache: bool = True,
    output_dir: Optional[str] = None,
) -> str:
    """Process a single document with performance optimizations.

    Args:
        pdf_path: Path to the PDF file
        use_parallel: Whether to use parallel processing
        use_cache: Whether to use caching
        output_dir: Optional directory to save output

    Returns:
        Processed text

    """
    # Create optimized configuration
    config = OptimizedOCRConfig(
        # Performance settings
        parallel_processing=use_parallel,
        max_workers=4,  # Adjust based on your CPU
        use_cache=use_cache,
        cache_dir=".ocr_cache",
        # OCR settings
        pdf_renderer_dpi=300,
        ocr_language="eng",  # Use "eng+deu" for English and German
        # Output settings
        output_format="markdown",
    )

    # Create pipeline
    pipeline = OptimizedOCRPipeline(config)

    # Process document with timing
    start_time = time.time()
    logger.info(f"Processing document: {pdf_path}")

    result = pipeline.process_pdf(pdf_path)

    duration = time.time() - start_time
    logger.info(f"Processing completed in {duration:.2f} seconds")

    # Save output if requested
    if output_dir:
        output_path = Path(output_dir) / f"{Path(pdf_path).stem}_processed.md"
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        logger.info(f"Saved result to: {output_path}")

    return result


def batch_process_documents(
    pdf_dir: str,
    pattern: str = "*.pdf",
    batch_size: int = 5,
    max_workers: int = 4,
    output_dir: Optional[str] = None,
) -> None:
    """Process multiple documents in batches with performance optimizations.

    Args:
        pdf_dir: Directory containing PDF files
        pattern: Glob pattern to match PDF files
        batch_size: Number of documents to process in each batch
        max_workers: Maximum number of worker threads
        output_dir: Optional directory to save output

    """
    # Find all matching PDF files
    pdf_paths = glob.glob(os.path.join(pdf_dir, pattern))
    if not pdf_paths:
        logger.warning(f"No PDF files found matching pattern {pattern} in {pdf_dir}")
        return

    logger.info(f"Found {len(pdf_paths)} PDF files to process")

    # Create optimized configuration
    config = OptimizedOCRConfig(
        # Performance settings
        parallel_processing=True,
        max_workers=max_workers,
        use_cache=True,
        cache_dir=".ocr_cache",
        # Batch processing settings
        batched_processing=True,
        batch_size=batch_size,
        skip_processed_files=True,
        # OCR settings
        pdf_renderer_dpi=300,
        ocr_language="eng",
        # Output settings
        output_format="markdown",
    )

    # Create pipeline
    pipeline = OptimizedOCRPipeline(config)

    # Process all documents with timing
    start_time = time.time()

    results = pipeline.batch_process_pdfs(pdf_paths)

    duration = time.time() - start_time
    logger.info(f"Batch processing completed in {duration:.2f} seconds")
    logger.info(f"Average time per document: {duration / len(pdf_paths):.2f} seconds")

    # Save results if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for pdf_path, text in results.items():
            output_path = Path(output_dir) / f"{Path(pdf_path).stem}_processed.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
        logger.info(f"Saved all results to: {output_dir}")


def process_specific_pages(
    pdf_path: str,
    pages: List[int],
    use_cache: bool = True,
    output_dir: Optional[str] = None,
) -> str:
    """Process specific pages of a document with caching.

    Args:
        pdf_path: Path to the PDF file
        pages: List of page numbers to process (0-based)
        use_cache: Whether to use caching
        output_dir: Optional directory to save output

    Returns:
        Processed text

    """
    # Create pipeline
    config = OptimizedOCRConfig(
        use_cache=use_cache,
        cache_dir=".ocr_cache",
        output_format="markdown",
    )
    pipeline = OptimizedOCRPipeline(config)

    # Process specific pages
    logger.info(f"Processing pages {pages} from {pdf_path}")
    result = pipeline.process_pdf(pdf_path, pages=pages)

    # Save output if requested
    if output_dir:
        page_str = "-".join(str(p) for p in pages)
        output_path = Path(output_dir) / f"{Path(pdf_path).stem}_pages_{page_str}.md"
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        logger.info(f"Saved result to: {output_path}")

    return result


def main():
    """Run the optimization examples from command line."""
    parser = argparse.ArgumentParser(description="OCR Pipeline Optimization Examples")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single document processing
    single_parser = subparsers.add_parser("single", help="Process a single document")
    single_parser.add_argument("pdf_path", help="Path to the PDF file")
    single_parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    single_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    single_parser.add_argument("--output-dir", help="Directory to save output")

    # Batch processing
    batch_parser = subparsers.add_parser("batch", help="Process multiple documents")
    batch_parser.add_argument("pdf_dir", help="Directory containing PDF files")
    batch_parser.add_argument("--pattern", default="*.pdf", help="Glob pattern to match PDF files")
    batch_parser.add_argument("--batch-size", type=int, default=5, help="Number of documents in each batch")
    batch_parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads")
    batch_parser.add_argument("--output-dir", help="Directory to save output")

    # Specific pages processing
    pages_parser = subparsers.add_parser("pages", help="Process specific pages")
    pages_parser.add_argument("pdf_path", help="Path to the PDF file")
    pages_parser.add_argument("--pages", required=True, help="Comma-separated list of pages (0-based)")
    pages_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    pages_parser.add_argument("--output-dir", help="Directory to save output")

    args = parser.parse_args()

    if args.command == "single":
        process_single_document(
            args.pdf_path,
            use_parallel=not args.no_parallel,
            use_cache=not args.no_cache,
            output_dir=args.output_dir,
        )
    elif args.command == "batch":
        batch_process_documents(
            args.pdf_dir,
            pattern=args.pattern,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            output_dir=args.output_dir,
        )
    elif args.command == "pages":
        pages = [int(p.strip()) for p in args.pages.split(",")]
        process_specific_pages(
            args.pdf_path,
            pages=pages,
            use_cache=not args.no_cache,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
