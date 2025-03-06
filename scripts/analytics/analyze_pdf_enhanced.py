#!/usr/bin/env python3
"""Enhanced PDF analysis script for identifying tables and images in PDF docs.

This script can be integrated with a RAG system to improve document processing.
"""

import argparse
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

import fitz
import tabula


class PDFAnalyzer:
    """Class for analyzing PDF documents to identify tables and images."""

    def __init__(self, verbose: bool = False):
        """Initialize the PDF analyzer.

        Args:
            verbose: Whether to print verbose output during analysis.

        """
        self.verbose = verbose

    def analyze_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a PDF file to identify tables and images.

        Args:
            pdf_path: Path to the PDF file to analyze.
            output_dir: Directory to save the analysis results to.
                If None, saves to the same directory as the PDF.

        Returns:
            A dictionary containing the analysis results.

        """
        pdf_path = os.path.abspath(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Extract text from PDF using pdftotext
        try:
            text = self._extract_text_with_pdftotext(pdf_path)
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
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

            # Find image references in the page
            page_images = self._find_images(lines, page_number)
            images.extend(page_images)

            # Add page data
            page_data.append(
                {
                    "page_number": page_number,
                    "text_length": len(page_text),
                    "line_count": len(lines),
                    "table_count": len(page_tables),
                    "image_reference_count": len(page_images),
                }
            )

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

        # Save results to file if output_dir is specified
        if output_dir:
            self._save_results(result, pdf_path, output_dir)
        else:
            # Save to the same directory as the PDF
            self._save_results(result, pdf_path)

        # Print summary
        self._print_summary(result)

        return result

    def _extract_text_with_pdftotext(self, pdf_path: str) -> str:
        """Extract text from PDF using pdftotext command line tool."""
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", pdf_path, "-"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"pdftotext failed: {e.stderr}") from e
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
                images.append({"page": page_number, "line": i + 1, "text": line})

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

                # Start a new block or continue the current one
                if current_block is None:
                    current_block = {"page": page_number, "start_line": j + 1, "end_line": j + 1, "text": line}
                else:
                    current_block["end_line"] = j + 1
                    current_block["text"] += "\n" + line

            # Add the last block if it exists
            if current_block is not None:
                text_blocks.append(current_block)

        return text_blocks

    def _save_results(self, result: Dict[str, Any], pdf_path: str, output_dir: Optional[str] = None) -> None:
        """Save analysis results to a JSON file.

        Args:
            result: The analysis results.
            pdf_path: Path to the PDF file.
            output_dir: Directory to save the results to.
                If None, saves to the same directory as the PDF.

        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(pdf_path) + ".json")
        else:
            output_path = pdf_path + ".json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"Analysis results saved to {output_path}")

    def _print_summary(self, result: Dict[str, Any]) -> None:
        """Print a summary of the analysis results.

        Args:
            result: The analysis results.

        """
        if not self.verbose:
            return

        print(f"\nAnalysis of {result['filename']}:")
        print(f"  Pages: {result['page_count']}")
        print(f"  Tables: {len(result['tables'])}")
        print(f"  Image references: {len(result['images'])}")
        print(f"  Text blocks: {len(result['text_blocks'])}")

        # Print table information
        if result["tables"]:
            print("\nTables:")
            for i, table in enumerate(result["tables"], 1):
                text = table["text"].split("\n")[0]
                print(f"{i}. Page {table['page']}, Lines {table['start_line']}-{table['end_line']}:")
                print(f"   {text}")

        # Print image information
        if result["images"]:
            print("\nImage references:")
            for i, image in enumerate(result["images"], 1):
                text = image["text"]
                print(f"{i}. Page {image['page']}, Line {image['line']}: {text}")

    def _extract_tables_with_tabula(self, pdf_path: str) -> List[Dict]:
        """Extract tables from PDF using tabula-py."""
        try:
            # Use tabula-py to extract tables
            tables = tabula.read_pdf(
                pdf_path,
                pages="all",
                multiple_tables=True,
                guess=True,
                lattice=True,
            )

            # Convert tables to dictionaries
            result = []
            for i, table in enumerate(tables):
                if not table.empty:
                    result.append(
                        {
                            "table_id": i,
                            "page": i + 1,  # Approximate page number
                            "data": table.to_dict("records"),
                            "headers": table.columns.tolist(),
                        }
                    )
            return result
        except Exception as e:
            print(f"Error extracting tables with tabula: {e}")
            return []

    def _extract_images_with_pymupdf(self, pdf_path: str) -> List[Dict]:
        """Extract images from PDF using PyMuPDF."""
        try:
            # Open the PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            images = []

            for page_num, page in enumerate(doc):
                image_list = page.get_images(full=True)

                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)

                    if base_image:
                        images.append(
                            {
                                "image_id": f"{page_num + 1}_{img_idx + 1}",
                                "page": page_num + 1,
                                "width": base_image["width"],
                                "height": base_image["height"],
                                "format": base_image["ext"],
                            }
                        )

            return images
        except Exception as e:
            print(f"Error extracting images with PyMuPDF: {e}")
            return []


def analyze_directory(
    directory: str, output_dir: Optional[str] = None, verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Analyze all PDF files in a directory.

    Args:
        directory: Path to the directory containing PDF files.
        output_dir: Directory to save the analysis results to.
        verbose: Whether to print verbose output during analysis.

    Returns:
        A dictionary mapping filenames to analysis results.

    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Directory not found: {directory}")

    analyzer = PDFAnalyzer(verbose=verbose)
    results = {}

    # Find all PDF files in the directory
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return results

    # Analyze each PDF file
    for pdf_file in pdf_files:
        try:
            result = analyzer.analyze_pdf(pdf_file, output_dir)
            results[os.path.basename(pdf_file)] = result
        except Exception as e:
            print(f"Error analyzing {pdf_file}: {e}")
            results[os.path.basename(pdf_file)] = {"error": str(e)}

    return results


def main():
    """Execute the main PDF analysis function."""
    parser = argparse.ArgumentParser(description="Analyze PDF files to identify tables and images.")
    parser.add_argument("input_path", help="Path to a PDF file or directory containing PDF files.")
    parser.add_argument("-o", "--output-dir", help="Directory to save the analysis results to.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output during analysis.")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)

    if os.path.isdir(input_path):
        analyze_directory(input_path, args.output_dir, args.verbose)
    elif os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        analyzer = PDFAnalyzer(verbose=args.verbose)
        analyzer.analyze_pdf(input_path, args.output_dir)
    else:
        print("Input path must be a PDF file or a directory containing PDF files.")
        return 1

    return 0


if __name__ == "__main__":
    main()
