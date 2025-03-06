import json
import os
import re
import subprocess
import sys


def analyze_pdf(pdf_path):
    """Analyze a PDF file to identify tables, images, and structure using pdftotext."""
    results = {"filename": os.path.basename(pdf_path), "pages": [], "tables": [], "images": [], "text_blocks": []}

    try:
        # Extract text using pdftotext (part of poppler-utils)
        cmd = ["pdftotext", "-layout", pdf_path, "-"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            results["error"] = f"pdftotext failed: {stderr.decode('utf-8', errors='ignore')}"
            return results

        text = stdout.decode("utf-8", errors="ignore")

        # Split text into pages
        page_separator = "\f"  # Form feed character used by pdftotext to separate pages
        pages = text.split(page_separator)

        results["page_count"] = len(pages)

        # Analyze each page
        for page_num, page_text in enumerate(pages):
            if not page_text.strip():
                continue

            page_info = {
                "page_number": page_num + 1,
                "text_length": len(page_text),
                "line_count": page_text.count("\n"),
            }

            # Split page into lines
            lines = page_text.split("\n")

            # Look for potential tables (lines with multiple spaces or tab characters)
            table_markers = [
                "|",
                "\t",
                "  ",
                "+---",
                "+===",
                "----",
                "====",
                "Table",
                "Tabelle",
                "Tab.",
                "Tbl.",
                "Übersicht",
            ]

            # Find table regions
            in_table = False
            table_start = 0
            table_lines = []

            for i, line in enumerate(lines):
                # Check if line looks like it's part of a table
                is_table_line = any(marker in line for marker in table_markers)

                # Check for table headers/captions
                is_table_header = any(
                    re.search(rf"\b{re.escape(marker)}\b", line.lower())
                    for marker in ["table", "tabelle", "tab.", "tbl.", "übersicht"]
                )

                if is_table_line or is_table_header:
                    if not in_table:
                        in_table = True
                        table_start = i
                        table_lines = [line]
                    else:
                        table_lines.append(line)
                elif in_table:
                    # If we've collected at least 3 lines, consider it a table
                    if len(table_lines) >= 3:
                        table_text = "\n".join(table_lines)
                        results["tables"].append(
                            {
                                "page": page_num + 1,
                                "start_line": table_start + 1,
                                "end_line": i,
                                "line_count": len(table_lines),
                                "text": table_text[:200] + "..." if len(table_text) > 200 else table_text,
                            }
                        )
                    in_table = False
                    table_lines = []

            # Check for any remaining table at the end of the page
            if in_table and len(table_lines) >= 3:
                table_text = "\n".join(table_lines)
                results["tables"].append(
                    {
                        "page": page_num + 1,
                        "start_line": table_start + 1,
                        "end_line": len(lines),
                        "line_count": len(table_lines),
                        "text": table_text[:200] + "..." if len(table_text) > 200 else table_text,
                    }
                )

            # Look for image references
            image_markers = ["figure", "abbildung", "abb.", "fig.", "bild", "grafik"]
            for i, line in enumerate(lines):
                if any(re.search(rf"\b{re.escape(marker)}\b", line.lower()) for marker in image_markers):
                    results["images"].append(
                        {"page": page_num + 1, "line": i + 1, "text": line[:100] + "..." if len(line) > 100 else line}
                    )

            # Add page info
            page_info["table_count"] = sum(1 for t in results["tables"] if t["page"] == page_num + 1)
            page_info["image_reference_count"] = sum(1 for img in results["images"] if img["page"] == page_num + 1)
            results["pages"].append(page_info)

        # Summary statistics
        results["summary"] = {
            "total_pages": results["page_count"],
            "total_tables": len(results["tables"]),
            "total_image_references": len(results["images"]),
        }

    except Exception as e:
        results["error"] = str(e)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_pdf_simple.py <pdf_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    results = analyze_pdf(pdf_path)

    if "error" in results:
        print(f"Error: {results['error']}")
        sys.exit(1)

    # Print summary
    print(f"\nAnalysis of {results['filename']}:")
    print(f"Pages: {results['summary']['total_pages']}")
    print(f"Potential tables: {results['summary']['total_tables']}")
    print(f"Image references: {results['summary']['total_image_references']}")

    # Print table information
    print("\nPotential Tables:")
    for i, table in enumerate(results["tables"][:10]):  # Show first 10
        print(f"{i + 1}. Page {table['page']}, Lines {table['start_line']}-{table['end_line']}:")
        print(f"   {table['text'].replace(chr(10), ' ')[:80]}...")

    if len(results["tables"]) > 10:
        print(f"... and {len(results['tables']) - 10} more tables")

    # Print image information
    print("\nImage References:")
    for i, img in enumerate(results["images"][:10]):  # Show first 10
        print(f"{i + 1}. Page {img['page']}, Line {img['line']}: {img['text']}")

    if len(results["images"]) > 10:
        print(f"... and {len(results['images']) - 10} more image references")

    # Save detailed results to JSON
    output_file = f"{os.path.splitext(pdf_path)[0]}_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed analysis saved to {output_file}")
