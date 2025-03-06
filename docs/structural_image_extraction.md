# Structural Image Extraction

This document describes the structural image extraction feature added to the RAG/LLM system to better distinguish images from text in PDF documents.

## Overview

The structural image extraction feature uses PyMuPDF (fitz) to extract actual images from PDF documents based on the PDF structure, rather than relying solely on text references to images. This provides a more accurate way to identify and process images in PDFs.

## Benefits

- **More Accurate Image Detection**: Identifies actual images in the PDF structure, not just text references
- **Position Information**: Captures the position of images on the page
- **Surrounding Text**: Attempts to extract text surrounding the image, which may include captions
- **Image Dimensions**: Provides width and height information for images
- **Fallback Mechanism**: Falls back to text-based detection if structural extraction fails

## Requirements

To use the structural image extraction feature, you need to install PyMuPDF:

```bash
pip install pymupdf
```

## Usage

### In Python Code

```python
from scripts.analytics.pdf_extractor import PDFStructureExtractor

# Create an extractor with structural image extraction enabled
extractor = PDFStructureExtractor(
    output_dir="output",
    save_images=True,
    use_structural_image_extraction=True,
    verbose=True
)

# Extract from a PDF
result = extractor.extract_from_pdf("path/to/document.pdf")

# Access the extracted images
images = result.get('images', [])
for image in images:
    print(f"Image on page {image['page']}")
    if 'position' in image:
        print(f"Position: {image['position']}")
    if 'width' in image and 'height' in image:
        print(f"Dimensions: {image['width']}x{image['height']}")
    if 'surrounding_text' in image:
        print(f"Surrounding text: {image['surrounding_text']}")
```

### Using the Test Script

A test script is provided to demonstrate the structural image extraction capability:

```bash
python -m scripts.test_structural_image_extraction path/to/document.pdf --compare --verbose
```

Options:

- `--output-dir DIR`: Directory to save output files
- `--save-images`: Save extracted images to disk
- `--compare`: Compare structural extraction with text-based extraction
- `--verbose`: Print verbose output

### Using with the Retrieval Test Script

The `test_retrieval.py` script has been updated to support structural image extraction:

```bash
python -m scripts.test_retrieval --extract-images --use-enhanced-extraction --use-structural-image-extraction
```

## Integration with RAG

The structural image extraction is integrated with the RAG system through the `EnhancedPDFProcessor` class. When enabled, it:

1. Extracts actual images from PDFs using PyMuPDF
2. Creates document chunks for each image with metadata indicating it's a structural image
3. Includes position and dimension information in the metadata
4. Uses surrounding text as the content for the image document chunk

## Future Enhancements

Potential future enhancements to the structural image extraction feature:

1. **Image Content Extraction**: Extract and save the actual image content
2. **Image Classification**: Classify images as diagrams, charts, photographs, etc.
3. **OCR Integration**: Apply OCR to extract text from images
4. **Visual Feature Extraction**: Extract visual features for multimodal retrieval
5. **Image Embedding**: Generate embeddings for images to enable similarity search

## Troubleshooting

If you encounter issues with the structural image extraction:

- Ensure PyMuPDF is installed: `pip install pymupdf`
- Check if the PDF is compatible with PyMuPDF
- Try with `--verbose` to see detailed logs
- If structural extraction fails, the system will fall back to text-based extraction
