import json
import os
import sys

import fitz  # PyMuPDF


def analyze_pdf(pdf_path):
    """Analyze a PDF file to identify tables, images, and structure."""
    results = {"filename": os.path.basename(pdf_path), "pages": [], "tables": [], "images": [], "text_blocks": []}

    try:
        doc = fitz.open(pdf_path)

        # Basic document info
        results["page_count"] = len(doc)
        results["metadata"] = doc.metadata

        # Analyze each page
        for page_num, page in enumerate(doc):
            page_info = {
                "page_number": page_num + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "rotation": page.rotation,
            }

            # Extract text blocks
            text_blocks = page.get_text("blocks")
            page_blocks = []

            for block_num, block in enumerate(text_blocks):
                x0, y0, x1, y1, text, block_type, block_no = block

                # Check if this might be a table based on structure
                is_potential_table = False
                if "\n" in text and any(char in text for char in ["|", ",", ";", "\t"]):
                    is_potential_table = True

                # Check for table captions/references
                has_table_reference = any(
                    marker in text.lower() for marker in ["table", "tabelle", "tab.", "tbl.", "übersicht"]
                )

                block_info = {
                    "block_num": block_num,
                    "rect": [x0, y0, x1, y1],
                    "text_length": len(text),
                    "is_potential_table": is_potential_table,
                    "has_table_reference": has_table_reference,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                }
                page_blocks.append(block_info)

                # If it looks like a table, add to tables list
                if is_potential_table or has_table_reference:
                    results["tables"].append(
                        {
                            "page": page_num + 1,
                            "rect": [x0, y0, x1, y1],
                            "text_preview": text[:100] + "..." if len(text) > 100 else text,
                        }
                    )

            # Extract images
            img_list = page.get_images(full=True)
            page_images = []

            for img_num, img in enumerate(img_list):
                xref, _, _, _, _, _, _, _, _ = img
                base_image = doc.extract_image(xref)

                if base_image:
                    image_info = {
                        "image_num": img_num,
                        "xref": xref,
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "colorspace": base_image.get("colorspace", "unknown"),
                    }
                    page_images.append(image_info)

                    # Add to global images list
                    results["images"].append(
                        {
                            "page": page_num + 1,
                            "image_num": img_num,
                            "width": base_image["width"],
                            "height": base_image["height"],
                        }
                    )

            # Check for figure captions
            for block in text_blocks:
                _, _, _, _, text, _, _ = block
                if any(marker in text.lower() for marker in ["figure", "abbildung", "abb.", "fig.", "bild", "grafik"]):
                    results["images"].append(
                        {
                            "page": page_num + 1,
                            "is_caption": True,
                            "caption_text": text[:100] + "..." if len(text) > 100 else text,
                        }
                    )

            page_info["block_count"] = len(page_blocks)
            page_info["image_count"] = len(page_images)
            results["pages"].append(page_info)

            # Add text blocks to global list
            results["text_blocks"].extend(
                [
                    {
                        "page": page_num + 1,
                        "block_num": b["block_num"],
                        "is_potential_table": b["is_potential_table"],
                        "has_table_reference": b["has_table_reference"],
                        "text_preview": b["text_preview"],
                    }
                    for b in page_blocks
                ]
            )

        # Summary statistics
        results["summary"] = {
            "total_pages": len(doc),
            "total_tables": len(results["tables"]),
            "total_images": len(results["images"]),
            "total_text_blocks": len(results["text_blocks"]),
        }

    except Exception as e:
        results["error"] = str(e)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_pdf.py <pdf_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    results = analyze_pdf(pdf_path)

    # Print summary
    print(f"\nAnalysis of {results['filename']}:")
    print(f"Pages: {results['summary']['total_pages']}")
    print(f"Potential tables: {results['summary']['total_tables']}")
    print(f"Images: {results['summary']['total_images']}")
    print(f"Text blocks: {results['summary']['total_text_blocks']}")

    # Print table information
    print("\nPotential Tables:")
    for i, table in enumerate(results["tables"][:10]):  # Show first 10
        print(f"{i + 1}. Page {table['page']}: {table['text_preview']}")

    if len(results["tables"]) > 10:
        print(f"... and {len(results['tables']) - 10} more tables")

    # Print image information
    print("\nImages and Figure References:")
    for i, img in enumerate(results["images"][:10]):  # Show first 10
        if "is_caption" in img:
            print(f"{i + 1}. Page {img['page']} (Caption): {img['caption_text']}")
        else:
            print(f"{i + 1}. Page {img['page']}: {img['width']}x{img['height']} pixels")

    if len(results["images"]) > 10:
        print(f"... and {len(results['images']) - 10} more images")

    # Save detailed results to JSON
    output_file = f"{os.path.splitext(pdf_path)[0]}_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed analysis saved to {output_file}")
