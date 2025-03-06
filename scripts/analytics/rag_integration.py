#!/usr/bin/env python3
"""RAG Integration for PDF Analytics.

This module provides integration between the PDF analytics tools and the RAG system.
It enhances the document processing pipeline with improved table and image extraction.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.analytics.pdf_extractor import PDFStructureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnhancedPDFProcessor:
    """Enhanced PDF processor for RAG systems with improved table and image extraction."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        save_tables: bool = True,
        save_images: bool = False,
        verbose: bool = False,
    ):
        """Initialize the enhanced PDF processor.

        Args:
            output_dir: Directory to save extracted tables and images.
                If None, saves to a subdirectory of the PDF directory.
            save_tables: Whether to save extracted tables.
            save_images: Whether to save extracted images.
            verbose: Whether to print verbose output during extraction.

        """
        self.extractor = PDFStructureExtractor(
            output_dir=output_dir, save_tables=save_tables, save_images=save_images, verbose=verbose
        )
        self.output_dir = output_dir
        self.verbose = verbose

    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process a PDF file and extract structured content.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory to save extracted content.
                If None, uses the default output directory.

        Returns:
            A dictionary containing the extraction results and document chunks.

        """
        # Extract tables and images
        extraction_result = self.extractor.extract_from_pdf(pdf_path, output_dir or self.output_dir)

        # Create document chunks
        documents = self._create_document_chunks(extraction_result, pdf_path)

        # Add documents to the result
        extraction_result["documents"] = documents

        return extraction_result

    def _create_document_chunks(self, extraction_result: Dict[str, Any], pdf_path: str) -> List[Dict[str, Any]]:
        """Create document chunks from the extraction result.

        Args:
            extraction_result: The extraction result from PDFStructureExtractor.
            pdf_path: Path to the PDF file.

        Returns:
            A list of document chunks suitable for the RAG system.

        """
        documents = []

        # Add main text document
        if extraction_result.get("text_blocks"):
            for i, block in enumerate(extraction_result["text_blocks"]):
                documents.append(
                    {
                        "content": block["text"],
                        "metadata": {
                            "source": str(pdf_path),
                            "filename": os.path.basename(pdf_path),
                            "filetype": "pdf",
                            "content_type": "text",
                            "block_id": f"block_{i + 1}",
                            "page": block.get("page", 0),
                            "start_line": block.get("start_line", 0),
                            "end_line": block.get("end_line", 0),
                        },
                    }
                )

        # Add table documents
        if extraction_result.get("tables"):
            for _, table in enumerate(extraction_result["tables"]):
                # Get the saved path if available
                table_path = table.get("saved_path", "")
                table_content = table["text"]

                # If the table was saved as CSV, read the CSV content
                if table_path and os.path.exists(table_path):
                    try:
                        with open(table_path, "r", encoding="utf-8") as f:
                            table_content = f.read()
                    except Exception as e:
                        logger.warning(f"Error reading table file: {e}")

                documents.append(
                    {
                        "content": table_content,
                        "metadata": {
                            "source": str(pdf_path),
                            "filename": os.path.basename(pdf_path),
                            "filetype": "pdf",
                            "content_type": "table",
                            "table_id": table["table_id"],
                            "page": table["page"],
                            "start_line": table["start_line"],
                            "end_line": table["end_line"],
                            "table_path": (table_path if os.path.exists(table_path) else None),
                        },
                    }
                )

        # Add image documents
        if extraction_result.get("images"):
            for i, image in enumerate(extraction_result["images"]):
                documents.append(
                    {
                        "content": image.get("caption", f"Image on page {image['page']}"),
                        "metadata": {
                            "source": str(pdf_path),
                            "filename": os.path.basename(pdf_path),
                            "filetype": "pdf",
                            "content_type": "image",
                            "image_id": f"image_{image['page']}_{i + 1}",
                            "page": image["page"],
                            "start_line": image.get("start_line", 0),
                            "end_line": image.get("end_line", 0),
                        },
                    }
                )

        return documents


def process_directory(
    directory: str,
    output_dir: Optional[str] = None,
    save_tables: bool = True,
    save_images: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Process all PDF files in a directory.

    Args:
        directory: Directory containing PDF files.
        output_dir: Directory to save extracted content.
        save_tables: Whether to save extracted tables.
        save_images: Whether to save extracted images.
        verbose: Whether to print verbose output during extraction.

    Returns:
        A dictionary mapping file paths to extraction results.

    """
    processor = EnhancedPDFProcessor(
        output_dir=output_dir, save_tables=save_tables, save_images=save_images, verbose=verbose
    )

    results = {}
    pdf_files = list(Path(directory).glob("**/*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return results

    logger.info(f"Processing {len(pdf_files)} PDF files in {directory}")

    for pdf_file in pdf_files:
        pdf_path = str(pdf_file)
        logger.info(f"Processing {pdf_path}")

        try:
            result = processor.process_pdf(pdf_path, output_dir)
            results[pdf_path] = result
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            results[pdf_path] = {"error": str(e)}

    return results
