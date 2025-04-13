# OCR Module for Document Processing

## Overview

This module provides optical character recognition (OCR) capabilities for the LLM RAG system, specifically designed to extract text from PDF documents that contain images or scanned content. It consists of two main components:

1. **PDFImageConverter**: Converts PDF pages to high-resolution images
2. **TesseractOCREngine**: Performs OCR on images to extract text

The module is optimized for German language documents by default, but can be configured for other languages supported by Tesseract.

## Requirements

- Python 3.12+
- PyMuPDF (fitz)
- Tesseract OCR (needs to be installed on your system)
- pytesseract
- Pillow (PIL)

## Installation

Ensure Tesseract OCR is installed on your system:

### macOS:

```bash
brew install tesseract
brew install tesseract-lang  # For additional language packs
```

### Ubuntu/Debian:

```bash
apt-get install tesseract-ocr
apt-get install tesseract-ocr-deu  # For German language support
```

### Windows:

Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

## Usage Example

```python
from pathlib import Path
from llm_rag.document_processing.ocr import PDFImageConverter, TesseractOCREngine
from llm_rag.document_processing.ocr.pdf_converter import PDFImageConverterConfig
from llm_rag.document_processing.ocr.ocr_engine import TesseractConfig, PageSegmentationMode

# 1. Configure and create components
pdf_config = PDFImageConverterConfig(
    dpi=300,  # Higher DPI for better OCR quality
    use_alpha_channel=False
)

ocr_config = TesseractConfig(
    languages=["deu"],  # German language
    psm=PageSegmentationMode.AUTO,  # Let Tesseract determine page segmentation
    timeout=60  # Longer timeout for larger documents
)

pdf_converter = PDFImageConverter(config=pdf_config)
ocr_engine = TesseractOCREngine(config=ocr_config)

# 2. Process a PDF document
pdf_path = Path("path/to/document.pdf")
extracted_text = []

# Process each page
for page_num, image in pdf_converter.convert_pdf_to_images(pdf_path):
    # Perform OCR on the page image
    text = ocr_engine.image_to_text(image)
    extracted_text.append(text)
    print(f"Processed page {page_num+1} - extracted {len(text)} characters")

# 3. Combine the extracted text
full_text = "\n\n".join(extracted_text)
print(f"Total extracted text: {len(full_text)} characters")

# You can now use this text for further processing in the RAG pipeline
```

## Advanced Features

### Processing Specific Pages

```python
# Process only pages 5-10 (1-indexed)
pdf_config = PDFImageConverterConfig(
    dpi=300,
    first_page=5,
    last_page=10
)
```

### Using Different Output Formats

```python
from llm_rag.document_processing.ocr.ocr_engine import OCROutputFormat

# Get detailed OCR data with confidence scores
ocr_data = ocr_engine.image_to_data(
    image,
    output_format=OCROutputFormat.JSON
)

# For each recognized word
for i in range(len(ocr_data['text'])):
    word = ocr_data['text'][i]
    confidence = ocr_data['conf'][i]
    if confidence > 90:
        print(f"High confidence word: {word} ({confidence}%)")
```

## Integration with Document Processing Pipeline

This OCR module is designed to integrate with the broader document processing pipeline in the LLM RAG system. After extracting text from PDFs, you can:

1. Pass the text to chunking mechanisms
2. Generate embeddings for vector storage
3. Use for retrieval in RAG workflows

For complete workflow examples, see the main documentation.
