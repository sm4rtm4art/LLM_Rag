"""PDF Analytics Package.

This package provides tools for analyzing PDF documents, extracting tables and images,
and preparing them for integration with RAG systems.
"""

from scripts.analytics.analyze_pdf_enhanced import analyze_pdf as analyze_pdf_enhanced
from scripts.analytics.analyze_pdf_simple import analyze_pdf as analyze_pdf_simple
from scripts.analytics.pdf_extractor import PDFStructureExtractor, extract_from_directory

__all__ = ["PDFStructureExtractor", "extract_from_directory", "analyze_pdf_simple", "analyze_pdf_enhanced"]
