"""OCR (Optical Character Recognition) module for document processing.

This module provides components for extracting text from PDF documents
using OCR technology, primarily through:
1. PDF to image conversion
2. OCR processing on the extracted images

The primary classes are:
- PDFImageConverter: Converts PDF pages to high-resolution images
- TesseractOCREngine: Performs OCR on images to extract text
"""

from llm_rag.document_processing.ocr.ocr_engine import TesseractOCREngine
from llm_rag.document_processing.ocr.pdf_converter import PDFImageConverter

__all__ = ["PDFImageConverter", "TesseractOCREngine"]
