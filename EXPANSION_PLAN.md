# OCR Pipeline Expansion Plan

This document outlines the plan for integrating an OCR pipeline into the document processing system, focusing on extracting high-quality, structured text from various PDF documents.

## Guiding Principles

1.  **MVP First**: Deliver core functionality quickly and iterate.
2.  **Modular Design**: Build independent, testable components.
3.  **Incremental LLM Integration**: Introduce LLMs strategically after core OCR is stable.
4.  **Configuration Driven**: Allow easy configuration of engines, models, and parameters.
5.  **Robust Testing**: Implement comprehensive testing at each phase.

## Phase 1: Core OCR Engine & Text PDF Handling (MVP)

- **Goal**: Process text-based PDFs, extract raw text using OCR.
- **Tasks**:
  - [x] Create `src/llm_rag/document_processing/ocr/` directory structure.
  - [x] Implement `pdf_converter.py` using PyMuPDF to render PDF pages as images.
  - [x] Implement `ocr_engine.py` as a basic wrapper for Tesseract.
  - [ ] Create initial `pipeline.py` to orchestrate PDF -> Image -> OCR Text flow.
  - [x] Add basic configuration for Tesseract path and language models.
  - [x] Write unit tests for `pdf_converter` and `ocr_engine`.
  - [ ] Write basic integration test for the text PDF pipeline.
- **Outcome**: Ability to extract raw text from simple, text-based PDFs via the new OCR pipeline.

## Phase 2: Scanned/Image PDF Handling & Basic Structuring

- **Goal**: Handle scanned PDFs and produce basic structured output (Markdown).
- **Tasks**:
  - [ ] Enhance `pdf_converter.py` with image preprocessing options (deskewing, thresholding).
  - [ ] Implement `output_formatter.py` to convert raw OCR text into simple Markdown (page breaks, basic paragraphs).
  - [ ] Update `pipeline.py` to include preprocessing and formatting steps.
  - [ ] Create evaluation dataset with scanned PDFs.
  - [ ] Implement initial evaluation harness (CER/WER calculation).
  - [ ] Test pipeline thoroughly with various scanned document qualities.
  - [ ] Write unit tests for `output_formatter` and preprocessing functions.
- **Outcome**: Ability to extract semi-structured Markdown from both text and scanned PDFs.

## Phase 3: LLM-Based Cleaning

- **Goal**: Improve the quality and structure of the OCR output using a local LLM.
- **Tasks**:
  - [ ] Implement `llm_processor.py` with an `LLMCleaner` class/function.
  - [ ] Integrate `LLMCleaner` with `ModelFactory` (e.g., Gemma).
  - [ ] Develop and iterate on prompts for cleaning OCR errors and basic formatting (headings, lists).
  - [ ] Update `pipeline.py` to optionally call `LLMCleaner` after the `output_formatter`.
  - [ ] Evaluate quality improvement vs. processing time trade-off.
  - [ ] Add configuration for LLM model selection and cleaning parameters.
  - [ ] Write unit tests for `LLMCleaner` (mocking the LLM).
  - [ ] Enhance evaluation harness to include semantic similarity metrics.
- **Outcome**: Higher-fidelity Markdown output, potentially correcting OCR errors and improving structure.

## Phase 4: LLM-Based Verification (Optional)

- **Goal**: Add an optional layer of verification using a potentially different/secondary LLM.
- **Tasks**:
  - [ ] Implement `LLMVerifier` within `llm_processor.py`.
  - [ ] Design and iterate on prompts for verification tasks (fact-checking against image snippets, consistency checks).
  - [ ] Update `pipeline.py` to integrate `LLMVerifier` as an optional final step.
  - [ ] Evaluate the effectiveness and necessity based on Phase 3 results.
  - [ ] Add configuration for verification LLM and parameters.
  - [ ] Write unit tests for `LLMVerifier`.
- **Outcome**: Increased confidence in the extracted information's accuracy.

## Phase 5: Advanced Structuring & Optimization

- **Goal**: Enhance output structure (complex Markdown, JSON), optimize performance.
- **Tasks**:
  - [ ] Improve `output_formatter` and `LLMCleaner` for complex structures (tables, nested lists).
  - [ ] Add option for JSON output format.
  - [ ] Implement parallel processing for pages/documents (`concurrent.futures`).
  - [ ] Optimize image handling and conversion.
  - [ ] Evaluate alternative OCR engines or fine-tuning Tesseract if needed.
  - [ ] Investigate LLM inference optimization (quantization, batching).
  - [ ] Refine error handling and reporting throughout the pipeline.
  - [ ] Write tests for advanced structure extraction and JSON output.
  - [ ] Benchmark performance before and after optimizations.
- **Outcome**: Highly structured, accurate output (Markdown/JSON) suitable for diverse downstream applications, with improved processing speed.

## Testing & Evaluation Strategy

- **Dataset**:
  - [ ] Curate a diverse PDF dataset (text, scanned, complex layouts, tables, images).
  - [ ] Create corresponding "golden standard" Markdown/JSON outputs manually.
  - [ ] **CI Testing Data**: Create small, synthetic test PDFs that can be committed to the repository for CI testing, or implement robust mocking strategies for tests that run in CI environments without real test data.
- **Metrics**:
  - [ ] Implement Character Error Rate (CER) / Word Error Rate (WER) calculation.
  - [ ] Explore semantic similarity metrics (BLEU, ROUGE, embedding distance).
  - [ ] Define metrics for structural accuracy (e.g., table detection rate).
  - [ ] Incorporate manual review scoring system.
- **Methodology**:
  - [ ] Enforce unit tests for all modules.
  - [ ] Develop integration tests for component interactions.
  - [ ] Build and maintain an evaluation harness script to run the full pipeline on the dataset and report metrics.
  - [ ] Track metrics across phases to demonstrate improvement.

## Optimization & Scalability Considerations

- [ ] **Image Processing**: Systematically test image resolution and preprocessing effects.
- [ ] **OCR Engine**: Benchmark Tesseract vs. alternatives (EasyOCR, Cloud APIs) if performance/accuracy demands increase.
- [ ] **Parallelization**: Profile and implement parallel execution for bottlenecks (OCR, LLM).
- [ ] **LLM Inference**: Apply quantization, explore optimized serving frameworks (vLLM, TGI) if needed.
- [ ] **Caching**: Implement caching for intermediate results (images, raw OCR) to speed up development/re-runs.

## Potential Pitfalls & Mitigation

- [ ] **Poor OCR Quality**: Use extensive preprocessing, tune OCR parameters, rely more on LLM cleaning (with caution), flag documents for manual review.
- [ ] **LLM Inaccuracy**: Employ rigorous prompt engineering, temperature control, verification step, allow skipping LLM steps, monitor for introduced errors.
- [ ] **Performance Bottlenecks**: Address proactively with optimization techniques (parallelization, efficient models), manage user expectations.
- [ ] **Complex Structure Extraction**: Start simple (Markdown), use capable LLMs, combine with heuristics, consider specialized models if necessary.
- [ ] **Dependency Management**: Use Docker for environment consistency, maintain clear dependency lists (`requirements.txt`/`pyproject.toml`).
- [ ] **CI/Test Environment Discrepancies**: Ensure tests gracefully handle missing test data in CI environments by using synthetic data, mocks, or conditional test skipping. Add markers like `@pytest.mark.requires_test_data` to clearly identify tests that need real documents.

---

## Part 2: Document Comparison Feature

### Phase 6: Section Alignment & Embedding-Based Similarity (Comparison MVP)

- **Goal**: Compare two processed documents (output from Phase 3+) section-by-section using embeddings, identifying major similarities/differences. Output basic annotated Markdown.
- **Prerequisite**: Reliable structured text output (Markdown) from the core OCR pipeline (Phase 3+).
- **Tasks**:
  - [ ] **`document_parser.py`**: Develop a module to parse the structured output (Markdown/JSON) from the OCR pipeline into logical sections/paragraphs (e.g., based on headings, semantic breaks, or fixed chunks).
  - [ ] **Alignment Logic**: Implement a basic strategy to align corresponding sections/paragraphs between the two documents (e.g., matching headings, sequence alignment on section titles/content).
  - [ ] **`comparison_engine.py` (Embeddings)**:
    - Integrate an embedding model (reuse from RAG or choose a suitable one like Sentence-BERT).
    - Generate embeddings for each aligned section pair.
    - Calculate cosine similarity between embeddings.
  - [ ] **`diff_formatter.py` (Basic)**:
    - Implement logic to classify aligned sections based on similarity thresholds (e.g., `SIMILAR`, `DIFFERENT`, `NEW`, `DELETED`).
    - Generate a basic annotated Markdown output highlighting these classifications (e.g., `[DIFFERENT] \\n --- \\n Section A text \\n +++ \\n Section B text \\n`).
  - [ ] **Orchestration**: Update the main pipeline/API to accept two documents, run them through the OCR pipeline (if not already processed), parse, align, compare (embeddings), and format the diff.
  - [ ] **Testing**: Create test cases with known document pairs (identical, slightly modified, significantly different) and evaluate alignment and similarity scoring.
- **Outcome**: Ability to generate a basic Markdown diff indicating sections that are likely similar or different based on embedding similarity.

### Phase 7: LLM-Based Semantic Comparison & Refined Output

- **Goal**: Enhance comparison accuracy by using an LLM to analyze nuanced differences, identify semantic rewrites, and provide richer annotations.
- **Prerequisite**: Phase 6 MVP working.
- **Tasks**:
  - [ ] **`comparison_engine.py` (LLM Integration)**:
    - Integrate a local LLM (e.g., Gemma, Mistral) capable of comparing text snippets.
    - Use the LLM to analyze section pairs flagged as `DIFFERENT` by embeddings, or pairs with high embedding similarity but significant text difference (potential rewrites).
    - **Prompt Engineering**: Design prompts for the LLM to:
      - Compare meaning: "Do these two paragraphs convey the same core meaning? Explain the difference if not."
      - Identify rewrites: "Is the second paragraph a rewrite of the first, maintaining the meaning but changing the wording significantly?"
      - Assess structural changes within sections.
  - [ ] **`diff_formatter.py` (Enhanced)**:
    - Incorporate LLM analysis results into annotations (e.g., `[MODIFIED MEANING]`, `[REWRITTEN - SIMILAR MEANING]`, `[STRUCTURAL CHANGE]`).
    - Generate more granular diffs (e.g., using standard diff library output within annotations).
    - Explore adding HTML output for side-by-side views with color highlighting.
  - [ ] **Handling Noise/Layout**: Develop strategies within the comparison logic or LLM prompts to be robust to minor residual OCR noise (e.g., instructing the LLM to ignore minor typos if focusing on meaning). Abstracting layout via structured text helps, but LLM can potentially identify if structure _within_ a section differs significantly.
  - [ ] **Evaluation**: Develop metrics for comparison quality (e.g., accuracy in classifying change type, precision/recall on identifying specific differences, human evaluation of diff readability).
  - [ ] **Refined Alignment**: Improve section alignment logic using more advanced sequence alignment algorithms if needed, possibly guided by embedding similarity.
- **Outcome**: Detailed, semantically aware comparison highlighting not just _that_ sections differ, but _how_ they differ (meaning, rewrite, structure), presented in a clear format.

## Comparison Implementation Details & Strategies

1.  **Document Comparison Logic**:

    - **MVP (Phase 6):**
      - Parse structured text (Markdown/JSON) into sections/paragraphs.
      - Align sections (heading matching, sequence).
      - Generate embeddings for aligned pairs.
      - Use cosine similarity with thresholds (e.g., >0.95 = Similar, <0.7 = Different, intermediate = Check Needed).
    - **Enhancement (Phase 7):**
      - Feed pairs below a similarity threshold (e.g., <0.9) or pairs with high similarity (>0.9) but low lexical overlap (e.g., low BLEU/ROUGE score) to an LLM.
      - Prompt the LLM to classify the difference: Meaning Change, Rewrite (Same Meaning), Minor Edit, Structural Change.

2.  **Handling OCR Noise During Comparison**:

    - **Primary Strategy:** Rely on the LLM Cleaning step (Phase 3) to produce clean text _before_ comparison.
    - **Secondary Strategy (Phase 7):** Instruct the comparison LLM (via prompts) to be robust to minor variations typical of OCR noise when assessing semantic meaning. Embeddings are often inherently robust to minor noise.

3.  **Handling Different Layouts**:

    - **Core Strategy:** Compare the _structured text output_ (Markdown/JSON) generated by the OCR pipeline. This largely abstracts away the original visual layout.
    - **Alignment Robustness:** The section alignment logic needs to handle cases where sections might be reordered or split/merged. Start simple (sequence/headings), enhance with algorithms like Needleman-Wunsch if needed, potentially using embedding similarity to guide alignment.

4.  **Evaluating Semantic Similarity & Detecting Rewrites**:

    - **Embeddings (Phase 6):** Cosine similarity provides a good first pass for overall semantic similarity.
    - **LLMs (Phase 7):** Provide explicit prompts:
      - "Compare the core meaning of Text A and Text B. Are they semantically equivalent? Explain differences."
      - "Is Text B a rewrite of Text A, preserving the main ideas but using different phrasing? Describe the nature of the rewrite."
    - **Hybrid:** Use embeddings for speed, trigger LLM analysis for ambiguous cases or potential rewrites (high similarity + low lexical overlap).

5.  **Presenting the Comparison**:
    - **MVP (Markdown - Phase 6):**
      - Use simple block annotations: `[SIMILAR]`, `[DIFFERENT]`, `[NEW]`, `[DELETED]`.
      - Show content for `DIFFERENT` sections using `---` (Doc A) and `+++` (Doc B).
    - **Enhanced (Markdown/HTML - Phase 7):**
      - More descriptive Markdown annotations based on LLM analysis: `[MODIFIED MEANING]`, `[REWRITTEN - SIMILAR MEANING]`, etc.
      - Include standard text diffs (e.g., from `difflib`) within sections marked as different.
      - Generate HTML for interactive side-by-side views with color highlighting for different change types.

## Proposed Architecture for Comparison Feature

Maintain modularity:

1.  **OCR Pipeline Service**: (Input: PDF path/bytes -> Output: Structured Text File Path/JSON/Markdown String)
    - Contains modules from Phases 1-5 (`pdf_converter`, `ocr_engine`, `output_formatter`, `llm_processor`).
2.  **Document Comparison Service**: (Input: Paths/Content of two structured documents -> Output: Comparison Report String/File)
    - **`Document Parser`**: Segments input documents.
    - **`Alignment Engine`**: Aligns sections between the two documents.
    - **`Comparison Engine`**:
      - Interfaces with **Embedding Model Service/Library**.
      - (Optionally) Interfaces with **Comparison LLM Service/Library** (could be the same `ModelFactory` as OCR).
      - Calculates similarity and/or performs LLM analysis on aligned pairs.
    - **`Diff Formatter`**: Generates the final report (Markdown/HTML).
3.  **API Gateway/Orchestrator**: Handles incoming requests (e.g., compare two PDFs), calls the OCR Pipeline for both (if needed), then calls the Document Comparison Service, and returns the result.
4.  **(Optional) Shared Services**:
    - **Model Service (`ModelFactory`)**: Centralized loading/access for embedding and generative LLMs.
    - **Cache**: Store results of OCR processing or even section embeddings.
    - **Task Queue (Celery/RQ)**: For handling long OCR or comparison jobs asynchronously.
