    Objective: Implement the initial components for the OCR pipeline as defined in Phase 1 of EXPANSION_PLAN.md.

    Context:
    - Reference EXPANSION_PLAN.md for Phase 1 tasks and goals.
    - Reference REFACTORING_PLAN.md for architectural context (modularity, existing utilities).
    - Strictly adhere to all rules defined in .cursor/rules/.cursorrules (Python 3.12, uv, FastAPI/Pydantic where applicable, PEP 8, meaningful names, docstrings, type hints, testing, exception handling).

    Tasks:
    1.  ✅ Create the directory structure: `src/llm_rag/document_processing/ocr/`.
    2.  ✅ Implement `src/llm_rag/document_processing/ocr/pdf_converter.py`:
        - ✅ Create a class (e.g., `PDFImageConverter`) responsible for taking a PDF file path.
        - ✅ Use the `PyMuPDF` library (`fitz`) to render each page into a high-resolution image (e.g., return a generator of PIL Image objects).
        - ✅ Include configuration options (e.g., DPI for rendering).
        - ✅ Implement error handling for file access and PDF processing issues, using utilities from `src/llm_rag/utils/errors.py` if appropriate.
        - ✅ Add logging using `src/llm_rag/utils/logging.py`.
        - ✅ Include comprehensive docstrings and type hints.
        - Write basic unit tests for this module (e.g., using a sample PDF and mocking file system interactions if needed).
    3.  ✅ Implement `src/llm_rag/document_processing/ocr/ocr_engine.py`:
        - ✅ Create a class (e.g., `TesseractOCREngine`) acting as a wrapper for Tesseract.
        - ✅ It should accept an image (e.g., PIL Image object) as input.
        - ✅ Use the `pytesseract` library to perform OCR and return the extracted raw text.
        - ✅ Include configuration options (e.g., Tesseract path, language models, page segmentation mode).
        - ✅ Implement error handling for Tesseract execution issues.
        - ✅ Add logging using `src/llm_rag/utils/logging.py`.
        - ✅ Include comprehensive docstrings and type hints.
        - Write basic unit tests for this module (e.g., mocking the `pytesseract.image_to_string` call).
    4.  ✅ Implement `src/llm_rag/document_processing/ocr/pipeline.py`:
        - ✅ Create a class (e.g., `OCRPipeline`) that orchestrates the PDF → Image → OCR Text flow.
        - ✅ Combine the functionality of `PDFImageConverter` and `TesseractOCREngine`.
        - ✅ Define a configuration dataclass (e.g., `OCRPipelineConfig`) that encapsulates settings for both components.
        - ✅ Implement methods to process a PDF document and return OCR text (for the entire document or specified pages).
        - ✅ Handle errors properly using the error handling utilities in `src/llm_rag/utils/errors.py`.
        - ✅ Add comprehensive logging using `src/llm_rag/utils/logging.py`.
        - ✅ Include proper type hints and docstrings.
        - Write unit and integration tests in `tests/document_processing/ocr/test_pipeline.py`.
    5.  ✅ Ensure all dependencies (`PyMuPDF`, `pytesseract`, `Pillow`) are added using `uv`.
    6.  ✅ Apply code formatting (e.g., using ruff and mypy) to ensure PEP 8 compliance.
    7.  ✅ Ensure CI pipeline correctly handles OCR tests:
        - Add Tesseract installation step to CI workflow
        - Implement conditional test skipping for tests requiring real PDF files
        - Add appropriate timeouts to prevent hanging tests
        - Use environment variables to detect CI environment

    Additional Completed Tasks:
    1. ✅ Implemented `src/llm_rag/document_processing/ocr/output_formatter.py` for Phase 2:
        - ✅ Created formatters to convert OCR text to Markdown
        - ✅ Implemented basic structure detection (headings, lists)
        - ✅ Added configuration options for formatting preferences
    2. ✅ Implemented `src/llm_rag/document_processing/ocr/llm_processor.py` for Phase 3:
        - ✅ Created LLMCleaner class for enhancing OCR text quality
        - ✅ Integrated with ModelFactory
        - ✅ Implemented error rate estimation for selective cleaning
        - ✅ Added configuration options for LLM model selection
        - ✅ Updated pipeline to optionally use LLM cleaning
        - ✅ **Initial Hallucination Safeguards:**
          - Store original text alongside cleaned version (e.g., within `ProcessedDocument`)
          - Implement basic change detection metrics (e.g., char/word change percentage)
          - Apply constrained prompting techniques (temperature control, focused instructions)
          - Ensure LLM cleaning is configurable (on/off)

    Deliverables:
    - New/modified Python files: `src/llm_rag/document_processing/ocr/__init__.py`, `pdf_converter.py`, `ocr_engine.py`, `pipeline.py`, `output_formatter.py`, `llm_processor.py`, `llm_cleaning_example.py`
    - Associated test files (e.g., `tests/document_processing/ocr/test_pdf_converter.py`, `tests/document_processing/ocr/test_ocr_engine.py`, `tests/document_processing/ocr/test_pipeline.py`).
    - Updates to dependency files managed by `uv`.
    - ✅ CI workflow improvements for handling OCR tests (.github/workflows/ci-cd.yml, .github/scripts/test.sh).
