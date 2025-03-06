#!/usr/bin/env python3
"""PDF Structure Extractor for RAG System.

This script extracts tables and images from PDF documents and prepares them
for integration with a RAG system. It can be used as a standalone script
or imported as a module.
"""

import argparse
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PDFStructureExtractor:
    """Extract tables and images from PDF documents for RAG systems."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        save_tables: bool = True,
        save_images: bool = False,
        verbose: bool = False,
        use_structural_image_extraction: bool = False,
    ):
        """Initialize the PDF structure extractor.

        Args:
            output_dir: Directory to save extracted tables and images.
                If None, saves to a subdirectory of the PDF directory.
            save_tables: Whether to save extracted tables.
            save_images: Whether to save extracted images.
            verbose: Whether to print verbose output during extraction.
            use_structural_image_extraction: Whether to use structural image extraction
                using PyMuPDF instead of text-based image reference detection.

        """
        self.output_dir = output_dir
        self.save_tables = save_tables
        self.save_images = save_images
        self.verbose = verbose
        self.use_structural_image_extraction = use_structural_image_extraction

    def extract_from_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Extract tables and images from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory to save extracted tables and images.
                If None, uses the default output directory.

        Returns:
            A dictionary containing the extraction results.

        """
        pdf_path = os.path.abspath(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting from PDF: {pdf_path}")

        # Extract text from PDF using pdftotext
        try:
            text = self._extract_text_with_pdftotext(pdf_path)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return {"error": str(e)}

        # Split text into pages
        pages = self._split_into_pages(text)

        # Analyze each page
        page_data = []
        tables = []
        images = []

        for i, page_text in enumerate(pages):
            page_number = i + 1
            lines = page_text.split("\n")

            # Find tables in the page
            page_tables = self._find_tables(lines, page_number)
            tables.extend(page_tables)

            # Find image references in the page (text-based approach)
            if not self.use_structural_image_extraction:
                page_images = self._find_images(lines, page_number)
                images.extend(page_images)

            # Add page data
            page_data.append(
                {
                    "page_number": page_number,
                    "text_length": len(page_text),
                    "line_count": len(lines),
                    "table_count": len(page_tables),
                    "image_reference_count": len(page_images) if not self.use_structural_image_extraction else 0,
                }
            )

        # If using structural image extraction, extract actual images from PDF
        if self.use_structural_image_extraction:
            try:
                structural_images = self._extract_structural_images(pdf_path)
                images.extend(structural_images)

                # Update page data with structural image counts
                image_counts = {}
                for img in structural_images:
                    page = img["page"]
                    image_counts[page] = image_counts.get(page, 0) + 1

                for page_info in page_data:
                    page_num = page_info["page_number"]
                    page_info["image_count"] = image_counts.get(page_num, 0)
            except Exception as e:
                logger.warning(f"Structural image extraction failed: {e}")
                logger.warning("Falling back to text-based image reference detection")
                # Fall back to text-based image detection
                for i, page_text in enumerate(pages):
                    page_number = i + 1
                    lines = page_text.split("\n")
                    page_images = self._find_images(lines, page_number)
                    images.extend(page_images)

                    # Update page data
                    page_data[i]["image_reference_count"] = len(page_images)

        # Create summary
        summary = {"total_pages": len(pages), "total_tables": len(tables), "total_image_references": len(images)}

        # Create result dictionary
        result = {
            "filename": os.path.basename(pdf_path),
            "page_count": len(pages),
            "pages": page_data,
            "tables": tables,
            "images": images,
            "summary": summary,
            "text_blocks": self._extract_text_blocks(pages),
        }

        # Save tables and images if requested
        if self.save_tables and tables:
            self._save_tables(result, pdf_path, output_dir or self.output_dir)

        if self.save_images and images:
            self._save_image_references(result, pdf_path, output_dir or self.output_dir)

        # Save analysis results
        self._save_analysis(result, pdf_path, output_dir or self.output_dir)

        # Print summary if verbose
        if self.verbose:
            self._print_summary(result)

        return result

    def _extract_text_with_pdftotext(self, pdf_path: str) -> str:
        """Extract text from PDF using pdftotext command-line tool."""
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", pdf_path, "-"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"pdftotext failed: {err.stderr}") from err
        except FileNotFoundError as err:
            raise RuntimeError("pdftotext not found. Please install poppler-utils.") from err

    def _split_into_pages(self, text: str) -> List[str]:
        """Split the extracted text into pages.

        Args:
            text: The extracted text from the PDF.

        Returns:
            A list of pages, where each page is a string.

        """
        # Split by form feed character
        pages = text.split("\f")
        # Remove empty pages
        return [page for page in pages if page.strip()]

    def _find_tables(self, lines: List[str], page_number: int) -> List[Dict[str, Any]]:
        """Find potential tables in a page.

        Args:
            lines: The lines of text in the page.
            page_number: The page number.

        Returns:
            A list of dictionaries containing table information.

        """
        tables = []
        current_table = None

        for i, line in enumerate(lines):
            # Table indicators: multiple spaces, alignment patterns, or tabular structure
            is_table_line = re.search(r"\s{3,}", line) and (
                re.search(r"[.]{3,}", line)
                or re.search(r"[|]", line)
                or re.search(r"\d+\s{2,}[A-Za-z]", line)
                or re.search(r"[A-Za-z]+\s{2,}\d+", line)
                or re.search(r"^\s*\d+\.\d+\s+", line)
                or re.search(r"^\s*[A-Za-z]\)\s+", line)
                or re.search(r"^\s*\d+\)\s+", line)
                or re.search(r"^\s*[•-]\s+", line)
            )

            # Check for table headers
            is_header = re.search(r"Table|Tabelle|Tableau", line, re.IGNORECASE) or re.search(
                r"^\s*Nr\.|^Key|^Légende|^Legende", line
            )

            if is_table_line or is_header:
                if current_table is None:
                    current_table = {
                        "page": page_number,
                        "start_line": i + 1,
                        "end_line": i + 1,
                        "line_count": 1,
                        "text": line,
                        "table_id": f"table_{page_number}_{i + 1}",
                    }
                else:
                    current_table["end_line"] = i + 1
                    current_table["line_count"] += 1
                    # Limit text to first few lines to avoid huge outputs
                    if current_table["line_count"] <= 5:
                        current_table["text"] += "\n" + line
                    elif current_table["line_count"] == 6:
                        current_table["text"] += "\n..."
            else:
                if current_table is not None:
                    # Only add tables with at least 2 lines
                    if current_table["line_count"] >= 2:
                        tables.append(current_table)
                    current_table = None

        # Add the last table if it exists
        if current_table is not None and current_table["line_count"] >= 2:
            tables.append(current_table)

        return tables

    def _find_images(self, lines: List[str], page_number: int) -> List[Dict[str, Any]]:
        """Find references to images in a page.

        Args:
            lines: The lines of text in the page.
            page_number: The page number.

        Returns:
            A list of dictionaries containing image reference information.

        """
        images = []

        for i, line in enumerate(lines):
            # Look for image references
            if re.search(r"Figure|Bild|Abbildung|Fig\.", line, re.IGNORECASE) or re.search(
                r"image|picture", line, re.IGNORECASE
            ):
                images.append(
                    {"page": page_number, "line": i + 1, "text": line, "image_id": f"image_{page_number}_{i + 1}"}
                )

        return images

    def _extract_text_blocks(self, pages: List[str]) -> List[Dict[str, Any]]:
        """Extract meaningful text blocks from pages.

        Args:
            pages: List of page texts.

        Returns:
            A list of dictionaries containing text block information.

        """
        text_blocks = []

        for i, page_text in enumerate(pages):
            page_number = i + 1
            lines = page_text.split("\n")

            current_block = None

            for j, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    if current_block is not None:
                        text_blocks.append(current_block)
                        current_block = None
                    continue

                # Skip lines that are likely part of tables
                if (
                    re.search(r"[.]{3,}", line)
                    or re.search(r"^\s*\d+\.\d+\s+", line)
                    or re.search(r"^\s*[A-Za-z]\)\s+", line)
                ):
                    if current_block is not None:
                        text_blocks.append(current_block)
                        current_block = None
                    continue

                # Start a new block or continue the current one
                if current_block is None:
                    current_block = {
                        "page": page_number,
                        "start_line": j + 1,
                        "end_line": j + 1,
                        "text": line,
                        "block_id": f"block_{page_number}_{j + 1}",
                    }
                else:
                    current_block["end_line"] = j + 1
                    current_block["text"] += " " + line

            # Add the last block if it exists
            if current_block is not None:
                text_blocks.append(current_block)

        return text_blocks

    def _save_tables(self, result: Dict[str, Any], pdf_path: str, output_dir: Optional[str] = None) -> None:
        """Save extracted tables to CSV files.

        Args:
            result: The extraction results.
            pdf_path: Path to the PDF file.
            output_dir: Directory to save the tables to.

        """
        if not result.get("tables"):
            logger.info("No tables to save")
            return

        pdf_name = os.path.basename(pdf_path).replace(".pdf", "")

        if output_dir:
            tables_dir = os.path.join(output_dir, "tables")
        else:
            pdf_dir = os.path.dirname(pdf_path)
            tables_dir = os.path.join(pdf_dir, f"{pdf_name}_tables")

        os.makedirs(tables_dir, exist_ok=True)

        logger.info(f"Saving {len(result['tables'])} tables to {tables_dir}")

        for table in result["tables"]:
            table_id = table["table_id"]
            table_path = os.path.join(tables_dir, f"{pdf_name}_{table_id}.csv")

            # Convert table text to CSV format
            csv_content = self._convert_table_to_csv(table["text"])

            with open(table_path, "w", encoding="utf-8") as f:
                f.write(csv_content)

            # Add path to table data
            table["saved_path"] = table_path

        logger.info(f"Tables saved to {tables_dir}")

    def _convert_table_to_csv(self, table_text: str) -> str:
        """Convert table text to CSV format.

        Args:
            table_text: The text of the table.

        Returns:
            The table in CSV format.

        """
        lines = table_text.split("\n")
        csv_lines = []

        for line in lines:
            # Replace multiple spaces with a single comma
            csv_line = re.sub(r"\s{2,}", ",", line.strip())
            csv_lines.append(csv_line)

        return "\n".join(csv_lines)

    def _save_image_references(self, result: Dict[str, Any], pdf_path: str, output_dir: Optional[str] = None) -> None:
        """Save image references to a JSON file.

        Args:
            result: The extraction results.
            pdf_path: Path to the PDF file.
            output_dir: Directory to save the image references to.

        """
        if not result.get("images"):
            logger.info("No image references to save")
            return

        pdf_name = os.path.basename(pdf_path).replace(".pdf", "")

        if output_dir:
            images_dir = os.path.join(output_dir, "images")
        else:
            pdf_dir = os.path.dirname(pdf_path)
            images_dir = os.path.join(pdf_dir, f"{pdf_name}_images")

        os.makedirs(images_dir, exist_ok=True)

        logger.info(f"Saving {len(result['images'])} image references to {images_dir}")

        # Save all image references to a single JSON file
        images_path = os.path.join(images_dir, f"{pdf_name}_images.json")

        with open(images_path, "w", encoding="utf-8") as f:
            json.dump(result["images"], f, indent=2, ensure_ascii=False)

        logger.info(f"Image references saved to {images_path}")

    def _save_analysis(self, result: Dict[str, Any], pdf_path: str, output_dir: Optional[str] = None) -> None:
        """Save the analysis results to a JSON file.

        Args:
            result: The analysis results.
            pdf_path: Path to the PDF file.
            output_dir: Directory to save the results to.

        """
        pdf_name = os.path.basename(pdf_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{pdf_name}_analysis.json")
        else:
            pdf_dir = os.path.dirname(pdf_path)
            output_path = os.path.join(pdf_dir, f"{pdf_name}_analysis.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Analysis saved to {output_path}")

    def _print_summary(self, result: Dict[str, Any]) -> None:
        """Print a summary of the extraction results.

        Args:
            result: The extraction results.

        """
        filename = result["filename"]
        summary = result["summary"]
        tables = result.get("tables", [])
        images = result.get("images", [])

        logger.info(f"Analysis of {filename}:")
        logger.info(f"Pages: {summary['total_pages']}")
        logger.info(f"Potential tables: {summary['total_tables']}")
        logger.info(f"Image references: {summary['total_image_references']}")

        if tables and self.verbose:
            logger.info("Potential Tables:")
            for i, table in enumerate(tables[:5], 1):
                text = table["text"].replace("\n", " ")
                if len(text) > 80:
                    text = text[:77] + "..."
                logger.info(f"{i}. Page {table['page']}, Lines {table['start_line']}-{table['end_line']}: {text}")

            if len(tables) > 5:
                logger.info(f"... and {len(tables) - 5} more tables")

        if images and self.verbose:
            logger.info("Image References:")
            for i, image in enumerate(images[:5], 1):
                text = image["text"].replace("\n", " ")
                if len(text) > 80:
                    text = text[:77] + "..."
                logger.info(f"{i}. Page {image['page']}, Line {image['line']}: {text}")

            if len(images) > 5:
                logger.info(f"... and {len(images) - 5} more image references")

    def _extract_structural_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract actual images from PDF using structural information.

        This method uses PyMuPDF (fitz) to extract images based on the PDF structure,
        which provides more accurate image detection than text-based approaches.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A list of dictionaries containing image information.

        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF (fitz) not found. Install with 'pip install pymupdf'")
            return []

        images = []

        try:
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc):
                # Get images as they appear in the PDF structure
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    xref = img[0]  # image reference

                    # Get basic image info without extracting the image data
                    # to keep the process lightweight
                    image_info = {
                        "page": page_num + 1,
                        "image_id": f"image_{page_num + 1}_{img_index + 1}",
                        "xref": xref,
                        "width": img[2],
                        "height": img[3],
                        "structural": True,
                    }

                    # Try to get position information
                    try:
                        rect = page.get_image_bbox(img)
                        if rect:
                            image_info["position"] = {"x1": rect.x0, "y1": rect.y0, "x2": rect.x2, "y2": rect.y2}
                    except Exception as e:
                        logger.debug(f"Could not get image position: {e}")

                    # Try to get surrounding text (potential caption)
                    try:
                        if "position" in image_info:
                            rect = fitz.Rect(
                                image_info["position"]["x1"],
                                image_info["position"]["y1"],
                                image_info["position"]["x2"],
                                image_info["position"]["y2"],
                            )
                            # Expand rect slightly to capture nearby text
                            # Use a method compatible with PyMuPDF
                            expanded_rect = rect + 20  # Add 20 points in all directions
                            text_around = page.get_text("text", clip=expanded_rect)
                            if text_around:
                                image_info["surrounding_text"] = text_around
                    except Exception as e:
                        logger.debug(f"Could not get surrounding text: {e}")

                    images.append(image_info)

            doc.close()

        except Exception as e:
            logger.warning(f"Error extracting structural images: {e}")
            return []

        return images


def extract_from_directory(
    directory: str,
    output_dir: Optional[str] = None,
    save_tables: bool = True,
    save_images: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Extract tables and images from all PDF files in a directory.

    Args:
        directory: Directory containing PDF files.
        output_dir: Directory to save extracted tables and images.
        save_tables: Whether to save extracted tables.
        save_images: Whether to save extracted images.
        verbose: Whether to print verbose output during extraction.

    Returns:
        A dictionary mapping filenames to extraction results.

    """
    extractor = PDFStructureExtractor(
        output_dir=output_dir, save_tables=save_tables, save_images=save_images, verbose=verbose
    )
    results = {}

    pdf_files = list(Path(directory).glob("**/*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return results

    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

    for pdf_file in pdf_files:
        logger.info(f"Processing {pdf_file}...")
        try:
            result = extractor.extract_from_pdf(str(pdf_file), output_dir)
            results[str(pdf_file)] = result
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
            results[str(pdf_file)] = {"error": str(e)}

    return results


def main():
    """Execute the main PDF extraction workflow."""
    parser = argparse.ArgumentParser(description="Extract tables and images from PDF documents for RAG systems.")
    parser.add_argument("input_path", help="Path to a PDF file or directory containing PDF files.")
    parser.add_argument("output_dir", help="Directory to save extracted content.")
    parser.add_argument(
        "--extract-tables", action="store_true", help="Extract tables from PDFs"
    )
    parser.add_argument(
        "--extract-images", action="store_true", help="Extract images from PDFs"
    )
    parser.add_argument(
        "--extract-text", action="store_true", help="Extract text from PDFs"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)

    if os.path.isdir(input_path):
        extract_from_directory(input_path, args.output_dir, args.extract_tables, args.extract_images, args.verbose)
    elif os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        extractor = PDFStructureExtractor(
            output_dir=args.output_dir,
            save_tables=args.extract_tables,
            save_images=args.extract_images,
            verbose=args.verbose
        )
        extractor.extract_from_pdf(input_path, args.output_dir)
    else:
        logger.error(f"Invalid input path: {input_path}")
        logger.error("Please provide a PDF file or a directory containing PDF files.")


if __name__ == "__main__":
    main()
