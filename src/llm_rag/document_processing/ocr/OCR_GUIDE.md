# OCR Pipeline Guide

This module provides a complete OCR (Optical Character Recognition) pipeline for converting PDF documents to structured text. It supports both scanned PDFs and digital text PDFs.

## Features

- **PDF to Image Conversion**: High-quality rendering of PDF documents to images
- **Image Preprocessing**: Options to enhance image quality for better OCR results
  - Deskewing (correcting tilted text)
  - Thresholding (improving contrast)
  - Contrast adjustment
  - Sharpening
  - Denoising
- **OCR Processing**: Using Tesseract OCR engine with configurable options
- **Output Formatting**: Converting raw OCR text to structured formats
  - Markdown formatting with automatic detection of headings, lists, etc.
  - JSON output (basic support)
  - Plain text output
- **Evaluation**: Metrics for evaluating OCR quality
  - Character Error Rate (CER)
  - Word Error Rate (WER)

## Usage

### Basic Usage

```python
from llm_rag.document_processing.ocr.pipeline import OCRPipeline

# Create pipeline with default settings
pipeline = OCRPipeline()

# Process a PDF and get text
text = pipeline.process_pdf("path/to/document.pdf")

# Process and save directly to Markdown
output_path = pipeline.save_to_markdown("path/to/document.pdf", "output/directory")
```

### Advanced Usage with Preprocessing

```python
from llm_rag.document_processing.ocr.pipeline import OCRPipeline, OCRPipelineConfig

# Create custom configuration with preprocessing
config = OCRPipelineConfig(
    # PDF settings
    pdf_dpi=400,  # Higher DPI for better quality

    # Preprocessing settings
    preprocessing_enabled=True,
    deskew_enabled=True,      # Fix tilted text
    threshold_enabled=True,   # Improve contrast
    threshold_method="adaptive",
    contrast_adjust=1.2,      # Slightly increase contrast
    sharpen_enabled=True,     # Make text sharper

    # OCR settings
    languages=["eng", "deu"],  # Process English and German text
    psm=3,                     # Page segmentation mode

    # Output formatting
    output_format="markdown",
    detect_headings=True,
    detect_lists=True
)

# Create pipeline with custom configuration
pipeline = OCRPipeline(config)

# Process PDF and save to file
output_path = pipeline.process_and_save(
    "path/to/document.pdf",
    "output/directory",
    format="markdown"  # Can also be "json" or "txt"
)
```

### Evaluating OCR Quality

```python
from llm_rag.document_processing.ocr.evaluation import calculate_metrics, compare_preprocessing_methods

# Evaluate OCR quality against ground truth
ground_truth = "This is the correct text content."
ocr_result = "Thls is the correot text contant."

# Calculate error rates
metrics = calculate_metrics(ground_truth, ocr_result)
print(f"Character Error Rate: {metrics['character_error_rate']:.4f}")
print(f"Word Error Rate: {metrics['word_error_rate']:.4f}")

# Compare results with and without preprocessing
base_ocr = "Thls is the correot text contant."
enhanced_ocr = "This is the correct text content."

comparison = compare_preprocessing_methods(
    ground_truth,
    base_ocr,
    enhanced_ocr
)

print(f"Improvement in CER: {comparison['improvement_percent']['character_error_rate']:.2f}%")
```

### Running the Example Script

The module includes an example script that demonstrates the OCR pipeline capabilities:

```bash
# Run with default settings on a sample document
python -m llm_rag.document_processing.ocr.example

# Process a specific PDF
python -m llm_rag.document_processing.ocr.example --pdf path/to/document.pdf

# Specify output directory
python -m llm_rag.document_processing.ocr.example --output output/directory

# Run only the basic or enhanced processing
python -m llm_rag.document_processing.ocr.example --mode basic
```

## Dependencies

- **PyMuPDF (fitz)**: For PDF rendering
- **Tesseract OCR**: For text extraction
- **OpenCV** (optional): For advanced image preprocessing
- **NumPy** and **Pillow**: For image processing

## Installation

Make sure Tesseract OCR is installed on your system:

- On macOS: `brew install tesseract`
- On Ubuntu: `sudo apt-get install tesseract-ocr`
- On Windows: Download and install from [Tesseract GitHub page](https://github.com/UB-Mannheim/tesseract/wiki)

Then install the required Python dependencies:

```bash
# Basic requirements
pip install pymupdf pillow pytesseract numpy

# For advanced preprocessing
pip install opencv-python
```
