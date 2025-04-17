# OCR Pipeline Improvements

This document summarizes the improvements made to the OCR pipeline to enhance performance, language handling, and overall usability.

## 1. Language Preservation and Translation

### Features Implemented:

- **Language Detection**: Automatically detect the language of input documents using `langdetect`
- **Language Preservation**: Ensure LLM doesn't translate content during OCR error correction
- **Translation Option**: Optional translation to specified target language
- **Language-Specific Models**: Support for language-specific LLM models for better results

### Files:

- `src/llm_rag/document_processing/ocr/llm_processor.py`: Enhanced LLMCleaner with language handling
- `src/llm_rag/document_processing/ocr/language_examples.py`: Usage examples
- `tests/document_processing/ocr/test_language_handling.py`: Unit tests

### Usage:

```python
# Language preservation example
cleaner_config = LLMCleanerConfig(
    preserve_language=True,  # Keep original language
    detect_language=True,    # Auto-detect language
)

# Translation example
cleaner_config = LLMCleanerConfig(
    translate_to_language="en",  # Translate to English
    detect_language=True,        # Auto-detect source language
)

# Language-specific model example
cleaner_config = LLMCleanerConfig(
    language_models={
        "de": "german-llm-model",  # Use specific model for German
        "fr": "french-llm-model",  # Use specific model for French
    }
)
```

## 2. Performance Optimization

### Features Implemented:

- **Parallel Processing**: Process multi-page documents in parallel
- **Caching**: Cache OCR results to avoid redundant processing
- **Batch Processing**: Process multiple documents efficiently
- **Incremental Processing**: Skip already processed files

### Files:

- `src/llm_rag/document_processing/ocr/optimized_pipeline.py`: Optimized OCR pipeline
- `src/llm_rag/document_processing/ocr/optimization_examples.py`: Usage examples
- `tests/document_processing/ocr/test_optimized_pipeline.py`: Unit tests

### Usage:

```python
# Parallel processing
config = OptimizedOCRConfig(
    parallel_processing=True,
    max_workers=4,
)

# Caching
config = OptimizedOCRConfig(
    use_cache=True,
    cache_dir=".ocr_cache",
    cache_ttl=7 * 24 * 60 * 60,  # 1 week
)

# Batch processing
pipeline = OptimizedOCRPipeline(config)
results = pipeline.batch_process_pdfs(["doc1.pdf", "doc2.pdf", "doc3.pdf"])
```

## 3. Future Improvements

Additional improvements that could be implemented in the future:

### Error Handling and Recovery

- More granular error reporting for OCR failures
- Retry logic for transient errors
- Better logging for performance metrics

### Output Formatting

- Enhanced Markdown formatter with better table detection
- HTML output formatter option
- Configurable page header/footer templates

### Additional Optimizations

- GPU acceleration for OCR processing
- Distributed processing for very large documents
- Adaptive DPI selection based on document quality

## Getting Started

1. Install the required dependencies:

```bash
pip install langdetect
```

2. Use the optimized OCR pipeline:

```python
from llm_rag.document_processing.ocr.optimized_pipeline import OptimizedOCRPipeline, OptimizedOCRConfig

config = OptimizedOCRConfig(
    parallel_processing=True,
    use_cache=True,
    preserve_language=True,
    detect_language=True,
)

pipeline = OptimizedOCRPipeline(config)
result = pipeline.process_pdf("document.pdf")
```

3. Run the example CLI:

```bash
python -m llm_rag.document_processing.ocr.optimization_examples single document.pdf
```
