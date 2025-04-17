Objective: Implement the document comparison feature as defined in Phase 6 of EXPANSION_PLAN.md, leveraging the S2 Chunking integration from Phase 5.5.

Context:

- Reference EXPANSION_PLAN.md Part 2: Document Comparison Feature for detailed requirements.
- This builds upon the OCR pipeline developed in earlier phases.
- Incorporate the S2 Chunking library (structural-semantic chunking) to improve document segmentation quality.
- The comparison module should operate on the structured and chunked output provided by the main OCR pipeline (using the chunking strategy configured in Phase 5.5, e.g., SemanticChunker initially).
- Strictly adhere to all rules defined in .cursor/rules/.cursorrules (Python 3.12, uv, FastAPI/Pydantic where applicable, PEP 8, meaningful names, docstrings, type hints, testing, exception handling).

Tasks:

1.  Create the directory structure: `src/llm_rag/document_processing/comparison/`.
2.  Implement `src/llm_rag/document_processing/comparison/document_parser.py`:
    - Create a class (e.g., `DocumentParser`) responsible for parsing structured output (Markdown/JSON) into logical sections/paragraphs.
    - Implement methods to segment documents based on headings, semantic breaks, or fixed chunks.
    - Handle different document formats (at minimum Markdown, with extensibility for JSON).
    - Include comprehensive error handling using utilities from `src/llm_rag/utils/errors.py`.
    - Add logging using `src/llm_rag/utils/logging.py`.
    - Include comprehensive docstrings and type hints.
    - Write unit tests for this module.
3.  Implement `src/llm_rag/document_processing/comparison/alignment.py`:
    - Create a class (e.g., `SectionAligner`) for aligning corresponding sections between two documents.
    - Implement matching strategies based on headings, content similarity, and section sequence.
    - Consider basic sequence alignment algorithms for section order.
    - Include configuration options for alignment thresholds and strategies.
    - Add proper error handling and logging.
    - Include comprehensive docstrings and type hints.
    - Write unit tests for various alignment scenarios.
4.  Implement `src/llm_rag/document_processing/comparison/comparison_engine.py`:
    - Create a class (e.g., `EmbeddingComparisonEngine`) to compare aligned sections using embeddings.
    - Integrate with an embedding model (reuse existing integration from the RAG system).
    - Implement methods to generate embeddings for sections and calculate cosine similarity.
    - Add classification logic with thresholds (e.g., `SIMILAR`, `DIFFERENT`, `NEW`, `DELETED`).
    - Include configuration options for embedding model selection and similarity thresholds.
    - Add proper error handling and logging.
    - Include comprehensive docstrings and type hints.
    - Write unit tests for this module.
5.  Implement `src/llm_rag/document_processing/comparison/diff_formatter.py`:
    - Create a class (e.g., `DiffFormatter`) to generate human-readable diff outputs.
    - Implement methods to format differences as annotated Markdown.
    - Include different formatting styles based on comparison classification.
    - Add configuration options for output format and detail level.
    - Add proper error handling and logging.
    - Include comprehensive docstrings and type hints.
    - Write unit tests for various formatting scenarios.
6.  Implement `src/llm_rag/document_processing/comparison/pipeline.py`:
    - Create a class (e.g., `ComparisonPipeline`) that orchestrates the entire comparison workflow.
    - Combine functionality of the parser, aligner, comparison engine, and formatter.
    - Create a configuration dataclass that encapsulates settings for all components.
    - Implement methods to compare two documents and generate a diff report.
    - Add comprehensive error handling and logging.
    - Include proper type hints and docstrings.
    - Write unit and integration tests for the complete pipeline.
7.  Update dependencies and ensure proper integration with the existing OCR pipeline.
8.  Apply code formatting (e.g., using black/ruff if configured) to ensure PEP 8 compliance.
9.  Implement CI-friendly testing:
    - Create small, synthetic test files that can be committed to the repository
    - For tests requiring larger PDF test data that cannot be committed due to copyright or size:
      - ✅ Add conditional test skipping using `pytest.mark.skipif` based on the presence of test files or CI environment
      - Implement robust mocking of file operations in tests to avoid dependencies on real files
      - Add clear documentation about which tests require local files versus which can run in CI
      - Consider separating fast and slow tests using pytest markers (`@pytest.mark.fast`, `@pytest.mark.slow`)
    - ✅ Ensure CI environment detects and installs necessary system dependencies like Tesseract OCR
    - ✅ Add appropriate timeouts to prevent tests from hanging indefinitely

Deliverables:

- New directory structure: `src/llm_rag/document_processing/comparison/`.
- New Python files: `__init__.py`, `document_parser.py`, `alignment.py`, `comparison_engine.py`, `diff_formatter.py`, `pipeline.py`.
- Associated test files in the `tests/document_processing/comparison/` directory.
- Updates to dependency files managed by `uv`.
- Documentation on how to use the comparison module (in docstrings and/or README).
- ✅ CI workflow improvements supporting both OCR and comparison module testing.

Expected Outcome:
A functional document comparison system that can:

1. Parse two structured documents (output from the OCR pipeline or other sources).
2. Align corresponding sections between the documents.
3. Compare aligned sections using embeddings to determine similarity.
4. Generate a human-readable diff report highlighting similarities, differences, additions, and deletions.
5. Be easily integrated with the existing OCR pipeline for end-to-end document processing and comparison.
