#!/usr/bin/env python3
"""Extract Tables from PDF Documents.

This script demonstrates how to use the PDF analytics package to extract
tables from PDF documents and save them as CSV files.
"""

import argparse
import logging
import os
from pathlib import Path

from scripts.analytics import PDFStructureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Extract tables from PDF documents and save as CSV files."""
    parser = argparse.ArgumentParser(description="Extract tables from PDF documents and save them as CSV files.")
    parser.add_argument("input_path", help="Path to a PDF file or directory containing PDF files.")
    parser.add_argument("output_dir", help="Directory to save extracted tables.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)

    # Create extractor
    extractor = PDFStructureExtractor(
        output_dir=args.output_dir, save_tables=True, save_images=True, verbose=args.verbose
    )

    if os.path.isdir(input_path):
        logger.info(f"Processing directory: {input_path}")
        pdf_files = list(Path(input_path).glob("**/*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {input_path}")
            return

        logger.info(f"Found {len(pdf_files)} PDF files")

        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file}...")
            try:
                extractor.extract_from_pdf(str(pdf_file))
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")

    elif os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        logger.info(f"Processing file: {input_path}")
        try:
            extractor.extract_from_pdf(input_path)
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")

    else:
        logger.error(f"Invalid input path: {input_path}")
        logger.error("Please provide a PDF file or a directory containing PDF files.")


if __name__ == "__main__":
    main()
