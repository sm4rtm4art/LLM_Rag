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
  - [x] Create initial `pipeline.py` to orchestrate PDF -> Image -> OCR Text flow.
  - [x] Add basic configuration for Tesseract path and language models.
  - [ ] Write unit tests for `pdf_converter` and `ocr_engine`.
  - [ ] Write basic integration test for the text PDF pipeline.
- **Outcome**: Ability to extract raw text from simple, text-based PDFs via the new OCR pipeline.

## Phase 2: Scanned/Image PDF Handling & Basic Structuring

- **Goal**: Handle scanned PDFs and produce basic structured output (Markdown).
- **Tasks**:
  - [x] Enhance `pdf_converter.py` with image preprocessing options (deskewing, thresholding).
  - [x] Implement `output_formatter.py` to convert raw OCR text into simple Markdown (page breaks, basic paragraphs).
  - [x] Update `pipeline.py` to include preprocessing and formatting steps.
  - [ ] Create evaluation dataset with scanned PDFs.
  - [ ] Implement initial evaluation harness (CER/WER calculation).
  - [ ] Test pipeline thoroughly with various scanned document qualities.
  - [ ] Write unit tests for `output_formatter` and preprocessing functions.
- **Outcome**: Ability to extract semi-structured Markdown from both text and scanned PDFs.

## Phase 3: LLM-Based Cleaning

- **Goal**: Improve the quality and structure of the OCR output using a local LLM.
- **Tasks**:
  - [x] Implement `llm_processor.py` with an `LLMCleaner` class/function.
  - [x] Integrate `LLMCleaner` with `ModelFactory` (e.g., Gemma).
  - [x] Develop and iterate on prompts for cleaning OCR errors and basic formatting (headings, lists).
  - [x] Update `pipeline.py` to optionally call `LLMCleaner` after the `output_formatter`.
  - [ ] Implement hallucination safeguards (see details in "LLM Hallucination Mitigation" section):
    - [x] **Initial:** Store original OCR text alongside cleaned version.
    - [x] **Initial:** Implement basic change detection metrics (e.g., character/word change percentage).
    - [x] **Initial:** Develop and apply constrained prompt techniques.
    - [ ] _Future:_ Implement more advanced safeguards (semantic drift detection, verification steps) as needed.
  - [x] Evaluate quality improvement vs. processing time trade-off.
  - [x] Add configuration for LLM model selection and cleaning parameters.
  - [ ] Write unit tests for `LLMCleaner` (mocking the LLM).
  - [ ] Enhance evaluation harness to include semantic similarity metrics.
- **Outcome**: Higher-fidelity Markdown output, potentially correcting OCR errors and improving structure.

## LLM Hallucination Mitigation

- **Goal**: Implement safeguards against LLM hallucinations in both OCR cleaning (Phase 3) and document comparison (Phase 7) to maintain text fidelity and trustworthiness. _Implementation will be incremental._
- **Tasks**:

  - [x] **Initial Safeguards**:
    - [x] **Storage:** Store original OCR text alongside LLM-processed text (e.g., in metadata or separate fields).
    - [x] **Basic Change Detection:** Implement simple metrics (e.g., character/word change percentage) between original and processed text. Log these metrics.
    - [x] **Thresholds:** Create configurable thresholds for acceptable change levels to flag potentially problematic edits.
    - [x] **Constrained Prompting:** Develop explicit instruction templates focusing on correction, not generation. Control parameters like temperature.
    - [x] **Configuration:** Ensure LLM steps can be easily enabled/disabled.
  - [ ] **_Future_ Enhancements**:
    - [ ] **Semantic Drift Detection:** Implement embedding-based similarity checks between original and cleaned text to catch meaning changes.
    - [ ] **Advanced Verification:** Implement secondary checks (e.g., different prompts, model cross-check, string similarity sanity checks). Consider checks against original PDF images for critical sections.
    - [ ] **Full Versioning System:** If needed, build a more robust system for tracking text lineage and accessing different versions via API.
    - [ ] **Human-in-the-Loop:** Develop flagging system and review interfaces if automated checks are insufficient.
    - [ ] **Hybrid Cleaning:** Implement rule-based pre-processing or staged LLM interventions if fine-grained control is required.

- **Outcome**: A system that leverages LLM capabilities while mitigating hallucination risks through foundational safeguards (prompting, basic change tracking) and offers a path for more advanced checks if needed.

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

## Phase 5.5: Advanced Document Chunking Integration

- **Goal**: Enhance document chunking by integrating an advanced, semantic-aware approach to improve the quality of document segments for RAG and document comparison, starting with LangChain's SemanticChunker.
- **Tasks**:

  - [ ] **LangChain SemanticChunker Integration (Initial Focus)**:
    - [ ] Implement `src/llm_rag/document_processing/semantic_chunker.py` as a wrapper for LangChain's SemanticChunker.
    - [ ] Configure the chunker to use appropriate embeddings for multilingual content.
    - [ ] Develop adapters to convert between our document format and LangChain's document formats.
    - [ ] Create configuration options for SemanticChunker-specific parameters (buffer size, similarity threshold).
    - [ ] Implement optimizations for handling large documents efficiently.
    - [ ] Evaluate effectiveness compared to basic chunking methods.
  - [ ] **Future Chunking Enhancements (Investigation)**:
    - [ ] **S2 Chunking:** Investigate integrating the S2 Chunking library (`s2-chunking-lib`) if layout-awareness becomes critical for specific document types (e.g., complex tables, forms). Requires adapting its YOLO model.
    - [ ] **Late Chunking:** Investigate implementing `late_chunking.py` if preserving very long-range context across the _entire_ document becomes necessary. Requires careful memory management and embedding strategy.
    - [ ] **Hybrid S2 + Late Chunking:** Explore combining structural boundaries from S2 with document-level embeddings from Late Chunking - particularly useful for documents with complex layouts but requiring semantic context preservation.
    - [ ] **Document Type Specialization:** Develop heuristics to apply S2 Chunking for visual documents and Late Chunking for long-form text documents like books.
    - [ ] **Chunking Strategy Selection (Simplified)**:
      - [ ] Initially, allow selection of the chunking strategy via configuration (e.g., "basic", "semantic").
      - [ ] _Future:_ Develop heuristics to automatically suggest or select the optimal strategy based on document properties if multiple strategies are implemented.
  - [ ] **Performance & Resource Management**:
    - [ ] Implement progressive chunking for very large documents if needed.
    - [ ] Add monitoring for embedding computation time and memory usage.
  - [ ] **Testing & Evaluation**:
    - [ ] Create test cases with representative German documents.
    - [ ] Implement basic error handling and logging for the chunking process.
    - [ ] Benchmark chunking quality metrics (start with retrieval simulation tests, basic coherence checks).
    - [ ] Compare SemanticChunker performance against baseline character/recursive chunking.

- **Outcome (Initial)**: A more semantically coherent chunking system using SemanticChunker, improving downstream RAG/comparison. Future iterations can incorporate layout-awareness (S2) or full-document context (Late Chunking) if necessary.

## Testing & Evaluation Strategy

- **Dataset**:
  - [ ] Curate a diverse PDF dataset (text, scanned, complex layouts, tables, images).
  - [ ] Create corresponding "golden standard" Markdown/JSON outputs manually.
  - [x] **CI Testing Data**: Create small, synthetic test PDFs that can be committed to the repository for CI testing, or implement robust mocking strategies for tests that run in CI environments without real test data.
- **Metrics**:
  - [ ] Implement Character Error Rate (CER) / Word Error Rate (WER) calculation.
  - [ ] Explore semantic similarity metrics (BLEU, ROUGE, embedding distance).
  - [ ] Define metrics for structural accuracy (e.g., table detection rate).
  - [ ] Incorporate manual review scoring system.
  - [ ] _Note:_ Advanced metrics like CLEval and specific chunking/comparison quality measures will be added incrementally as the relevant features mature.
- **Methodology**:
  - [x] Enforce unit tests for all modules.
  - [ ] Develop integration tests for component interactions.
  - [ ] Build and maintain an evaluation harness script to run the full pipeline on the dataset and report metrics.
  - [ ] Track metrics across phases to demonstrate improvement.

## Optimization & Scalability Considerations

- [x] **Image Processing**: Systematically test image resolution and preprocessing effects.
- [ ] **OCR Engine**: Benchmark Tesseract vs. alternatives (EasyOCR, Cloud APIs) if performance/accuracy demands increase.
- [x] **Parallelization**: Profile and implement parallel execution for bottlenecks (OCR, LLM).
- [ ] **LLM Inference**: Apply quantization, explore optimized serving frameworks (vLLM, TGI) if needed.
- [x] **Caching**: Implement caching for intermediate results (images, raw OCR) to speed up development/re-runs.

## Potential Pitfalls & Mitigation

- [ ] **Poor OCR Quality**: Use extensive preprocessing, tune OCR parameters, rely more on LLM cleaning (with caution), flag documents for manual review.
- [ ] **LLM Inaccuracy**: Employ rigorous prompt engineering, temperature control, verification step, allow skipping LLM steps, monitor for introduced errors.
- [ ] **Performance Bottlenecks**: Address proactively with optimization techniques (parallelization, efficient models), manage user expectations.
- [ ] **Complex Structure Extraction**: Start simple (Markdown), use capable LLMs, combine with heuristics, consider specialized models if necessary.
- [x] **Dependency Management**: Use Docker for environment consistency, maintain clear dependency lists (`requirements.txt`/`pyproject.toml`).
- [x] **CI/Test Environment Discrepancies**: Ensure tests gracefully handle missing test data in CI environments by using synthetic data, mocks, or conditional test skipping. Add markers like `@pytest.mark.requires_test_data` to clearly identify tests that need real documents.

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

### Phase 7: Advanced Semantic Comparison with LLM

- **Goal**: Enhance comparison accuracy by using an LLM for nuanced difference analysis, building upon the embedding-based similarity from Phase 6.
- **Prerequisite**: Phase 6 MVP working.
- **Tasks**:

  - [ ] **`comparison_engine.py` (LLM Integration)**:
    - [ ] Integrate a local LLM (e.g., Gemma, Mistral) capable of comparing text snippets.
    - [ ] Use the LLM to analyze section pairs flagged as `DIFFERENT` by embeddings, or pairs with high similarity (>0.9) but low lexical overlap (e.g., low BLEU/ROUGE score).
    - [ ] **Prompt Engineering**: Design prompts for the LLM to:
      - Compare meaning: "Do these two paragraphs convey the same core meaning? Explain the difference if not."
      - Identify rewrites: "Is the second paragraph a rewrite of the first, maintaining the meaning but changing the wording significantly?"
      - Assess structural changes within sections.
    - [ ] **Hallucination Mitigation for Comparison**:
      - [ ] **Initial:** Apply constrained prompting techniques specific to comparison.
      - [ ] **Initial:** Calculate and store confidence scores for LLM-identified differences (if model provides them, or via heuristics).
      - [ ] **Initial:** Use thresholds on confidence/change metrics to flag potentially unreliable comparisons.
    - [ ] **`diff_formatter.py` (Enhanced)**:
      - Incorporate LLM analysis results into annotations (e.g., `[MODIFIED MEANING]`, `[REWRITTEN - SIMILAR MEANING]`, `[STRUCTURAL CHANGE]`).
      - Generate more granular diffs (e.g., using standard diff library output within annotations).
      - Explore adding HTML output for side-by-side views with color highlighting.
      - Add metadata to indicate confidence level in each detected difference.
    - [ ] **Handling Noise/Layout**: Develop strategies within the comparison logic or LLM prompts to be robust to minor residual OCR noise. Structured text output helps abstract layout.
    - [ ] **Evaluation**: Develop metrics for comparison quality (e.g., accuracy in classifying change type, human evaluation of diff readability).
    - [ ] **Refined Alignment**: Improve section alignment logic using more advanced sequence alignment algorithms if needed, possibly guided by embedding similarity.
    - [ ] **Future Enhancement (Post-Phase 7):**
      - _If needed:_ Use ColBERT for token-level similarity assessment on ambiguous segments.

- **Outcome**: Detailed, semantically aware comparison highlighting not just _that_ sections differ, but _how_ they differ (meaning, rewrite, structure), presented in a clearer format. Token-level precision via ColBERT remains an option for future optimization.

## Enhanced Evaluation Metrics & Framework

- **Goal**: Implement a comprehensive evaluation framework with advanced metrics to assess OCR quality, chunking effectiveness, and comparison accuracy. _Implementation will be incremental._
- **Tasks**:

  - [ ] **OCR Quality Metrics**:
    - [ ] Implement Character Error Rate (CER) and Word Error Rate (WER) calculation.
    - [ ] _Future:_ Add Character-Level Evaluation (CLEval) for more granular assessment.
    - [ ] _Future:_ Create visualization tools for OCR error patterns.
    - [ ] _Future:_ Implement confidence scoring for OCR results.
  - [ ] **Chunking Quality Metrics**:
    - [ ] _Start with:_ Design retrieval simulation tests to assess chunk quality in RAG scenarios.
    - [ ] _Future:_ Develop structural coherence measures (broken paragraphs, lists, tables).
    - [ ] _Future:_ Implement semantic coherence measures using embedding similarity.
  - [ ] **Comparison Quality Metrics**:
    - [ ] _Start with:_ Design human evaluation protocols and scoring systems for diff quality/readability.
    - [ ] _Future:_ Implement BLEU and ROUGE metrics for lexical similarity.
    - [ ] _Future:_ Develop embedding-based similarity measures for semantic comparison.
    - [ ] _Future:_ Create precision/recall metrics for difference detection.
  - [ ] **Automated Evaluation Pipeline**:
    - [ ] Build a configurable evaluation pipeline (incrementally adding metrics).
    - [ ] Implement benchmark tracking.
    - [ ] Create basic reports, enhancing with visualizations later.
    - [ ] _Future:_ Design A/B testing framework.
  - [ ] **Ground Truth Dataset Creation**:
    - [ ] Establish guidelines for creating ground truth data.
    - [ ] Develop tools/processes to assist annotation.
    - [ ] Build a versioned repository of ground truth data.
    - [ ] _Future:_ Implement data augmentation techniques.

- **Outcome**: A robust evaluation framework providing insights into system performance, enabling data-driven decisions, and supporting regression testing. _Initial focus on core metrics, expanding over time._

## Modular Retrieval System Implementation

- **Goal**: Create a flexible, modular retrieval system that supports both document comparison and RAG applications with interchangeable components.
- **Tasks**:

  - **Core Retrieval Interface**:
    - [ ] Design a modular interface for retrieval components (`src/llm_rag/retrieval/interfaces.py`).
    - [ ] Implement base classes for query processors, retrievers, and re-rankers.
    - [ ] Define standardized input/output formats for each component to ensure compatibility.
    - [ ] Create a registry system for dynamically loading different implementations.
  - **Query Processing Modules**:
    - [ ] Implement various query transformation strategies:
      - [ ] Keyword extraction
      - [ ] Query expansion
      - [ ] Query decomposition for complex questions
      - [ ] Multilingual query handling
    - [ ] Develop query embedding services using different models.
    - [ ] Create plugin architecture for custom preprocessing steps.
  - **Retrieval Modules**:
    - [ ] Implement diverse retrieval strategies:
      - [ ] Dense retrieval (embedding-based)
      - [ ] Sparse retrieval (BM25, TF-IDF)
      - [ ] Hybrid approaches
      - [ ] _Future:_ ColBERT token-level retrieval (if integrated for comparison)
    - [ ] Create adapters for different vector stores (FAISS, Chroma, Milvus).
    - [ ] Implement caching mechanisms for frequent queries.
  - **Re-ranking Modules**:
    - [ ] Implement different re-ranking strategies:
      - [ ] Cross-encoder re-ranking
      - [ ] Fusion methods for combining multiple retrieval results
      - [ ] LLM-based contextual re-ranking
    - [ ] Create configuration system for re-ranking parameters.
    - [ ] Design feedback mechanisms to improve ranking over time.
  - **Pipeline Orchestration**:
    - [ ] Develop a pipeline builder for configuring retrieval workflows.
    - [ ] Implement dynamic selection logic based on query characteristics.
    - [ ] Create monitoring and logging components for pipeline performance.
    - [ ] Design fallback strategies for handling retrieval failures.
  - **Integration with OCR and Comparison Systems**:
    - [ ] Connect the retrieval system to the document storage/indexing system.
    - [ ] Implement specialized retrievers for document comparison use cases.
    - [ ] Create adapters between different document representations (OCR output, chunks, etc.).
    - [ ] Develop batch processing capabilities for multi-document comparison.

- **Outcome**: A highly adaptable retrieval system capable of supporting various use cases from simple keyword search to complex semantic comparison, with the ability to easily swap or upgrade components as requirements evolve or new technologies emerge.

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
    - **Future Enhancement (Post-Phase 7):**
      - _If needed:_ Use ColBERT for token-level similarity assessment on ambiguous segments.

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
    - **Future Enhancement (Post-Phase 7):**
      - _If needed:_ ColBERT token-level interactions provide fine-grained comparison, potentially effective for detecting rewrites and semantic-preserving edits.
    - **Hybrid:** Use embeddings for speed, trigger LLM analysis for ambiguous cases or potential rewrites.

5.  **Presenting the Comparison**:
    - **MVP (Markdown - Phase 6):**
      - Use simple block annotations: `[SIMILAR]`, `[DIFFERENT]`, `[NEW]`, `[DELETED]`.
      - Show content for `DIFFERENT` sections using `---` (Doc A) and `+++` (Doc B).
    - **Enhanced (Markdown/HTML - Phase 7):**
      - More descriptive Markdown annotations based on LLM analysis: `[MODIFIED MEANING]`, `[REWRITTEN - SIMILAR MEANING]`, etc.
      - Include standard text diffs (e.g., from `difflib`) within sections marked as different.
      - _Future:_ Integrate ColBERT token-level highlighting if implemented.
      - Generate HTML for interactive side-by-side views with color highlighting for different change types.

## Proposed Architecture for Document Processing System

Maintain modularity:

1.  **OCR Pipeline Service**: (Input: PDF path/bytes -> Output: Structured Text File Path/JSON/Markdown String)

    - Contains modules from Phases 1-5 (`pdf_converter`, `ocr_engine`, `output_formatter`, `llm_processor`).
    - **Advanced Chunking Subsystem** (Simplified initial focus):
      - `semantic_chunker.py`: Wrapper for LangChain's SemanticChunker (primary method).
      - `chunking_orchestrator.py`: Manages selection (initially via config).
      - _Future Investigation:_ `s2chunking_adapter.py`, `late_chunking.py`.
      - Produces semantically coherent chunks.

2.  **Document Comparison Service**: (Input: Paths/Content of two structured documents -> Output: Comparison Report String/File)

    - **`Document Parser`**: Segments input documents.
    - **`Alignment Engine`**: Aligns sections between the two documents.
    - **`Comparison Engine`**:
      - Interfaces with **Embedding Model Service/Library**.
      - (Optionally) Interfaces with **Comparison LLM Service/Library**.
      - _Future Investigation:_ Integration with **ColBERT**.
      - Calculates similarity and/or performs LLM analysis on aligned pairs.
    - **`Diff Formatter`**: Generates the final report (Markdown/HTML).

3.  **Modular Retrieval System**: (Input: Query and document collection -> Output: Relevant chunks/documents)

    - **`Query Processor`**: Transforms and embeds user queries.
    - **`Retriever`**: Implements various retrieval strategies (dense, sparse, hybrid). _Future: ColBERT_.
    - **`Re-ranker`**: Refines retrieval results using advanced ranking techniques.
    - **`Orchestrator`**: Manages the retrieval pipeline and component selection.
    - Leverages the Advanced Chunking Subsystem for higher-quality document segments.

4.  **API Gateway/Orchestrator**: Handles incoming requests (e.g., compare two PDFs), calls the OCR Pipeline for both (if needed), then calls the Document Comparison Service, and returns the result.

5.  **Shared Services**:
    - **Model Service (`ModelFactory`)**: Centralized loading/access for embedding and generative LLMs.
    - **Evaluation Framework**: Implements comprehensive metrics _incrementally_.
    - **Cache**: Store results of OCR processing or even section embeddings.
    - **Task Queue (Celery/RQ)**: For handling long OCR or comparison jobs asynchronously.

This simplified architecture focuses on delivering core advanced functionality first (SemanticChunker, LLM Comparison) while keeping more complex options (S2/Late Chunking, ColBERT) as potential future enhancements based on evaluation and need.
