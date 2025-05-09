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
  - ✅ Create `src/llm_rag/document_processing/ocr/` directory structure.
  - ✅ Implement `pdf_converter.py` using PyMuPDF to render PDF pages as images.
  - ✅ Implement `ocr_engine.py` as a basic wrapper for Tesseract.
  - ✅ Create initial `pipeline.py` to orchestrate PDF -> Image -> OCR Text flow.
  - ✅ Add basic configuration for Tesseract path and language models.
  - [ ] Write unit tests for `pdf_converter` and `ocr_engine`.
  - [ ] Write basic integration test for the text PDF pipeline.
- **Outcome**: Ability to extract raw text from simple, text-based PDFs via the new OCR pipeline.

## Phase 2: Scanned/Image PDF Handling & Basic Structuring

- **Goal**: Handle scanned PDFs and produce basic structured output (Markdown).
- **Tasks**:
  - ✅ Enhance `pdf_converter.py` with image preprocessing options (deskewing, thresholding).
  - ✅ Implement `output_formatter.py` to convert raw OCR text into simple Markdown (page breaks, basic paragraphs).
  - ✅ Update `pipeline.py` to include preprocessing and formatting steps.
  - [ ] Create evaluation dataset with scanned PDFs.
  - [ ] Implement initial evaluation harness (CER/WER calculation).
  - [ ] Test pipeline thoroughly with various scanned document qualities.
  - [ ] Write unit tests for `output_formatter` and preprocessing functions.
- **Outcome**: Ability to extract semi-structured Markdown from both text and scanned PDFs.

## Phase 3: LLM-Based Cleaning

- **Goal**: Improve the quality and structure of the OCR output using a local LLM.
- **Tasks**:
  - ✅ Implement `llm_processor.py` with an `LLMCleaner` class/function.
  - ✅ Integrate `LLMCleaner` with `ModelFactory` (e.g., Gemma).
  - ✅ Develop and iterate on prompts for cleaning OCR errors and basic formatting (headings, lists).
  - ✅ Update `pipeline.py` to optionally call `LLMCleaner` after the `output_formatter`.
  - [ ] Implement hallucination safeguards (see details in "LLM Hallucination Mitigation" section):
    - ✅ **Initial:** Store original OCR text alongside cleaned version.
    - ✅ **Initial:** Implement basic change detection metrics (e.g., character/word change percentage).
    - ✅ **Initial:** Develop and apply constrained prompt techniques.
    - [ ] _Future:_ Implement more advanced safeguards (semantic drift detection, verification steps) as needed.
  - ✅ Evaluate quality improvement vs. processing time trade-off.
  - ✅ Add configuration for LLM model selection and cleaning parameters.
  - [ ] Write unit tests for `LLMCleaner` (mocking the LLM).
  - [ ] Enhance evaluation harness to include semantic similarity metrics.
- **Outcome**: Higher-fidelity Markdown output, potentially correcting OCR errors and improving structure.

## LLM Hallucination Mitigation

- **Goal**: Implement safeguards against LLM hallucinations in both OCR cleaning (Phase 3) and document comparison (Phase 7) to maintain text fidelity and trustworthiness. _Implementation will be incremental._
- **Tasks**:

  - ✅ **Initial Safeguards**:
    - ✅ **Storage:** Store original OCR text alongside LLM-processed text (e.g., in metadata or separate fields).
    - ✅ **Basic Change Detection:** Implement simple metrics (e.g., character/word change percentage) between original and processed text. Log these metrics.
    - ✅ **Thresholds:** Create configurable thresholds for acceptable change levels to flag potentially problematic edits.
    - ✅ **Constrained Prompting:** Develop explicit instruction templates focusing on correction, not generation. Control parameters like temperature.
    - ✅ **Configuration:** Ensure LLM steps can be easily enabled/disabled.
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
  - ✅ **CI Testing Data**: Create small, synthetic test PDFs that can be committed to the repository for CI testing, or implement robust mocking strategies for tests that run in CI environments without real test data.
- **Metrics**:
  - [ ] Implement Character Error Rate (CER) / Word Error Rate (WER) calculation.
  - [ ] Explore semantic similarity metrics (BLEU, ROUGE, embedding distance).
  - [ ] Define metrics for structural accuracy (e.g., table detection rate).
  - [ ] Incorporate manual review scoring system.
  - [ ] _Note:_ Advanced metrics like CLEval and specific chunking/comparison quality measures will be added incrementally as the relevant features mature.
- **Methodology**:
  - ✅ Enforce unit tests for all modules.
  - [ ] Develop integration tests for component interactions.
  - [ ] Build and maintain an evaluation harness script to run the full pipeline on the dataset and report metrics.
  - [ ] Track metrics across phases to demonstrate improvement.

## Optimization & Scalability Considerations

- ✅ **Image Processing**: Systematically test image resolution and preprocessing effects.
- [ ] **OCR Engine**: Benchmark Tesseract vs. alternatives (EasyOCR, Cloud APIs) if performance/accuracy demands increase.
- ✅ **Parallelization**: Profile and implement parallel execution for bottlenecks (OCR, LLM).
- [ ] **LLM Inference**: Apply quantization, explore optimized serving frameworks (vLLM, TGI) if needed.
- ✅ **Caching**: Implement caching for intermediate results (images, raw OCR) to speed up development/re-runs.

## Potential Pitfalls & Mitigation

- [ ] **Poor OCR Quality**: Use extensive preprocessing, tune OCR parameters, rely more on LLM cleaning (with caution), flag documents for manual review.
- [ ] **LLM Inaccuracy**: Employ rigorous prompt engineering, temperature control, verification step, allow skipping LLM steps, monitor for introduced errors.
- [ ] **Performance Bottlenecks**: Address proactively with optimization techniques (parallelization, efficient models), manage user expectations.
- [ ] **Complex Structure Extraction**: Start simple (Markdown), use capable LLMs, combine with heuristics, consider specialized models if necessary.
- ✅ **Code Style Consistency**: Configure ruff to handle mixed indentation styles across modules:
  - ✅ Use per-file-ignores in pyproject.toml for specific modules (e.g., comparison module)
  - ✅ Add standard ignore rules for tab indentation (W191), tab docstrings (D206), and mixed indentation (E101)
  - ✅ Maintain consistent style within each module while allowing for module-specific conventions
- ✅ **Dependency Management**: Use Docker for environment consistency, maintain clear dependency lists (`requirements.txt`/`pyproject.toml`).
- ✅ **CI/Test Environment Discrepancies**: Ensure tests gracefully handle missing test data in CI environments by using synthetic data, mocks, or conditional test skipping. Add markers like `@pytest.mark.requires_test_data` to clearly identify tests that need real documents.

---

## Part 2: Document Comparison Feature

This section outlines the development of a feature to compare two documents, identify differences and similarities, and present them in a human-readable format. The initial MVP (Phase 6) focuses on establishing the pipeline with section alignment and embedding-based similarity. This serves as a crucial foundation for Phase 7, which introduces Large Language Models (LLMs) to perform advanced semantic analysis and provide nuanced explanations of the differences, representing the core objective of the comparison feature.

### Phase 6: Section Alignment & Embedding-Based Similarity (Comparison MVP)

- **Goal**: Compare two processed documents (output from Phase 3+) section-by-section using embeddings, identifying major similarities/differences. Output basic annotated Markdown. This phase provides the essential structure and first-pass analysis for the more advanced LLM-based comparison in Phase 7.
- **Prerequisite**: Reliable structured text output (Markdown) from the core OCR pipeline (Phase 3+).
- **Tasks**:
  - ✅ **`document_parser.py`**: Develop a module to parse the structured output (Markdown/JSON) from the OCR pipeline into logical sections/paragraphs (e.g., based on headings, semantic breaks, or fixed chunks). _(Status: Core Markdown parsing refactored with delegated strategy; Pydantic models implemented; List parsing verified; Setext heading parsing fixed)_
  - ✅ **`alignment.py`**:
    - ✅ Define core alignment interface and data structures
    - ✅ Implement alignment strategy selection (heading-based, content-based, structural)
    - ✅ Implement robust sequence alignment algorithms for document sections
    - ✅ Add configuration options for alignment thresholds and behavior
    - ✅ Fix unhashable type issue with Section objects in sets
    - ✅ Implement comprehensive tests covering alignment strategies
  - ✅ **`comparison_engine.py` (Embeddings)**:
    - ✅ Integrate an embedding model (mock implementation for now, actual model path configurable)
    - ✅ Generate embeddings for each aligned section pair
    - ✅ Calculate cosine similarity between embeddings
    - ✅ Implement comparison result classification
    - ✅ Create robust test suite for comparison methods
  - ✅ **`diff_formatter.py` (Basic)**:
    - ✅ Implement logic to classify aligned sections based on similarity thresholds
    - ✅ Generate annotated output in multiple formats (Markdown, HTML, Text)
    - ✅ Add configuration for output format and detail level
    - ✅ Create tests for different output formats and configurations
      - Initial tests for test_diff_formatter.py created and passing with 90% coverage for diff_formatter.py.
  - ✅ **`pipeline.py` (Orchestration)**:
    - ✅ Implement pipeline to process documents through parsing, alignment, comparison, and formatting
    - ✅ Add caching mechanism for intermediate results
    - ✅ Implement comprehensive test suite for pipeline workflow
    - [x] Complete integration with OCR pipeline for end-to-end workflow (Conceptual integration via Markdown confirmed, E2E test script `demos/document_comparison/run_ocr_comparison_e2e.py` created and updated for async LLM calls.)
  - [x] **Architectural Enhancement**: (Largely Addressed)
    - [x] Consider refactoring the comparison module to follow a more modular approach similar to the RAG pipeline (Achieved: centralized domain models, component protocols, updated components)
    - [x] Implement clear interfaces between components for better extensibility (Achieved: via `component_protocols.py`)
    - [x] Design for progressive enhancement with LLM capabilities (Foundation laid, initial LLM integration prototyped in Phase 7 section)
  - [ ] **S2 Chunking Integration**:
    - [/'] Clarify how S2 Chunking from Phase 5.5 will integrate with document*parser.py (`DocumentParser` ready for `List[InputChunk]` via `PRE_CHUNKED` format. Next: Determine \_where* chunking occurs and how `InputChunk` list is generated and passed to `ComparisonPipeline`.)
    - [x] Add adapter code to use chunked output for comparison (`DocumentParser`'s `parse_from_input_chunks` method serves as this adapter within the parser for `InputChunk` lists.)
    - [ ] Ensure consistent document representation between chunking and comparison
  - ✅ **Testing**: Create test cases with known document pairs (identical, slightly modified, significantly different) and evaluate alignment and similarity scoring.
    - ✅ Test all core components
      - test_comparison_engine.py and test_document_parser.py existing and passing.
      - test_diff_formatter.py created, and all its tests are passing.
    - ✅ Fix test issues and ensure 100% pass rate
      - All 32 tests in tests/document_processing/comparison/ are passing.
- **Outcome**: Ability to generate a basic Markdown diff indicating sections that are likely similar or different based on embedding similarity. **This phase is foundational for Phase 7, and its components are now capable of supporting initial LLM integration.**

### Phase 7: Advanced Semantic Comparison with LLM

- **Goal**: Enhance comparison accuracy by using an LLM for nuanced difference analysis, building upon the embedding-based similarity from Phase 6.
- **Prerequisite**: Phase 6 MVP working.
- **Status**: Prototyping Enhanced.
- **Strategy**: Initial development may leverage smaller, faster generative LLMs for rapid prototyping of prompt engineering and LLM response handling, with the option to scale to larger models for production quality.
- **Tasks**:

  - [x] **`comparison_engine.py` (LLM Integration Prototyped)**:

    - [x] Design modular architecture: `LLMComparer` component created for LLM interactions, used by `EmbeddingComparisonEngine`.
    - [ ] Create a strategy pattern for selecting comparison method based on requirements (Future refinement).
    - [x] Integrate a local LLM (e.g., Phi-3 via Ollama) capable of comparing text snippets (`OllamaClient` created).
    - [x] Use the LLM to analyze section pairs flagged as `DIFFERENT` or `MODIFIED` by embeddings (Implemented in `EmbeddingComparisonEngine`).
    - [x] **Prompt Engineering (Initial Draft)**: Develop first-pass prompts for the LLM to:
      - Compare meaning and categorize differences (Initial `DEFAULT_PROMPT_TEMPLATE` in `LLMComparer`).
    - [x] **Configuration**: Add `LLMComparerPipelineConfig` to `ComparisonPipelineConfig` for LLM settings. Update E2E script to use this.
    - [x] **Async Operations**: Update `ComparisonPipeline`, `EmbeddingComparisonEngine`, and relevant protocols for asynchronous LLM calls.

  - [ ] **🔍 Domain-Specific Prompting and Legal Awareness (NEW)**:

    - [ ] Update the `LLMComparer.DEFAULT_PROMPT_TEMPLATE` to handle legal/contractual documents.
    - [ ] Include semantic-ontological classifications relevant to legal texts (e.g., `SEMANTIC_REWRITE`, `ONTOLOGICAL_SUBSUMPTION`, `LEGAL_EFFECT_CHANGE`, `STRUCTURAL_REORDERING`, `DIFFERENT_CONCEPTS`).
    - [ ] Ensure model instructions account for legal effect, roles (e.g., creditor/debtor), and obligations.
    - [ ] Incorporate fallback categories such as `UNCERTAIN` and `NO_MEANINGFUL_CONTENT` for robustness against noise or OCR errors.

  - [x] **Refine LLM Comparison Logic & Output (Next Steps)**:
    - [x] **`diff_formatter.py` (Enhanced)**:
      - [/] Incorporate LLM analysis results (`LLMAnalysisResult.comparison_category`, `explanation`) into annotations (e.g., `[MODIFIED MEANING (LLM)]`, `[REWRITTEN - SIMILAR MEANING (LLM)]`). (Done for Markdown, HTML, Text)
      - [ ] Generate more granular diffs if useful (e.g., using standard diff library output within annotations).
      - [ ] Explore adding HTML output for side-by-side views with color highlighting.
      - [ ] Add metadata to indicate confidence level in each detected difference.
    - [x] **LLM Trigger Refinement**: Improve conditions for when to call `LLMComparer` (e.g., handle empty sections, consider lexical overlap vs. embedding similarity). (Implemented short-content skip)
    - [x] **Result Aggregation**: Determine how `LLMAnalysisResult.comparison_category` refines the final `SectionComparison.result_type`. (Decision: LLM is supplementary for now, result_type unchanged by LLM category; documented).
    - [ ] **Prompt Iteration**: Refine `LLMComparer` prompts based on testing for accuracy and edge case handling.
    - [ ] **Hallucination Mitigation for Comparison (Initial)**:
      - [ ] Apply constrained prompting techniques specific to comparison (Current prompt asks for JSON, specific categories).
      - [ ] Calculate and store confidence scores for LLM-identified differences (Pydantic model has field, LLM prompt mentions it as optional).
      - [ ] Use thresholds on confidence/change metrics to flag potentially unreliable comparisons (Future step after confidence is reliably extracted).
  - [ ] **Handling Noise/Layout**: Develop strategies within the comparison logic or LLM prompts to be robust to minor residual OCR noise. Structured text output helps abstract layout. (OCR LLM cleaning enabled in E2E script).
  - [/] **Evaluation**: Develop metrics for comparison quality (e.g., accuracy in classifying change type, human evaluation of diff readability), especially for LLM-enhanced results. (Added essential unit tests for LLM logic).
  - [ ] **Refined Alignment**: Improve section alignment logic using more advanced sequence alignment algorithms if needed, possibly guided by embedding similarity.
  - [ ] **Future Enhancement (Post-Phase 7):**

    - _If needed:_ Use ColBERT for token-level similarity assessment on ambiguous segments.

  - [ ] **📘 Future Legal LLM Options (Post-Phi/Gemma Prototyping) (NEW)**:
    - [ ] Evaluate specialized legal models (e.g., Legal-BERT, CaseLaw-BERT, GerDaL-BERT, or custom fine-tuned BERT/other architectures for EU/German civil law).
    - [ ] Investigate ColBERT (potentially as part of a JuristenRAG-style approach) as an alignment-sensitive reranker or for fine-grained similarity assessment in long legal contexts (contract clauses, definitions).
    - [ ] Consider RAG-based verification using a legal knowledge base (e.g., German BGB/contract law database via SPARQL or RDF, potentially explored within a JuristenRAG framework).

- **Outcome**: **Initial prototype of LLM-enhanced comparison complete and refined.** Detailed, semantically aware comparison highlighting not just _that_ sections differ, but _how_ they differ (meaning, rewrite, structure), presented in a clearer format. System can now make calls to an LLM (via Ollama) for specified section pairs, handles short content, and stores structured results. LLM details are displayed in reports. Core code quality improved with docstrings, formatting, and linting. Essential unit tests for LLM logic added. **Further refinement will target legal document specifics, prompt engineering, and more comprehensive testing.**

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
    - [ ] **RAGatouille Integration**:
      - [ ] Implement wrapper for RAGatouille with its ColBERT-based retrieval
      - [ ] Configure RAGatouille for German language support
        - [ ] Evaluate colbert-european-multilingual model for direct use
        - [ ] Explore fine-tuning options with German-MiniLM or Gelectra models
        - [ ] Implement German-specific preprocessing (lemmatization, stemming) for domain content
      - [ ] Integrate with document processing pipeline for high-quality retrievals
      - [ ] Benchmark RAGatouille against other retrieval methods
        - [ ] Implement evaluation metrics: Hit@k, MRR, and nDCG
        - [ ] Compare against BM25, FAISS (with cosine), Chroma/HNSW, and hybrid approaches
      - [ ] Create fallback mechanisms when RAGatouille might not be optimal
      - [ ] Design hybrid retrieval combining RAGatouille with other methods
        - [ ] Implement fast first-pass retrieval (BM25/dense) for top-N documents
        - [ ] Apply ColBERT reranking on this narrowed document set
        - [ ] Explore both LangChain's retriever composition and manual implementation
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

## RAGatouille Advanced Retrieval Integration

- **Goal**: Enhance retrieval quality by integrating RAGatouille's advanced ColBERT-based retrieval capabilities into the system.
- **Tasks**:

  - [ ] **Core Implementation**:
    - [ ] Create `src/llm_rag/retrieval/ragatouille_retriever.py` implementing the retrieval interface.
    - [ ] Implement adapter patterns to convert between RAGatouille's data formats and our system's.
    - [ ] Configure RAGatouille with appropriate models for German and multilingual content.
      - [ ] Evaluate and implement colbert-european-multilingual model
      - [ ] Explore fine-tuning pathways using German-MiniLM or Gelectra
      - [ ] Develop domain-specific optimization if needed
    - [ ] Integrate with the document chunking system to optimize chunk sizes for RAGatouille.
  - [ ] **German Language Support**:
    - [ ] Implement preprocessing pipeline specific to German text:
      - [ ] German lemmatization for improved token matching
      - [ ] Stemming customized for domain-specific content
      - [ ] Special handling for compound words common in German
      - [ ] Domain-specific vocabulary enhancement
    - [ ] Evaluate multilingual vs. German-specific model performance
    - [ ] Create specialized indexing configurations for German content
  - [ ] **Configuration & Optimization**:
    - [ ] Implement parameter management for RAGatouille's retrieval settings.
    - [ ] Develop batching strategies for efficient processing of large document collections.
    - [ ] Create caching mechanism for RAGatouille indices to speed up repeated queries.
    - [ ] Explore quantization options for deployed models to reduce resource requirements.
  - [ ] **Hybrid Approaches**:
    - [ ] Implement fusion methods combining RAGatouille with BM25/dense retrievers.
    - [ ] Design query routing logic to select optimal retriever based on query characteristics.
    - [ ] Create ensemble retrieval options that combine results from multiple retrievers.
    - [ ] Develop two-stage retrieval pipeline:
      - [ ] Fast first-pass retrieval using BM25/vector search to identify candidate set
      - [ ] Apply ColBERT via RAGatouille for precise reranking of top candidates
      - [ ] Support both LangChain's retriever composition and custom implementation
  - [ ] **Evaluation & Testing**:
    - [ ] Benchmark RAGatouille against baseline retrievers on domain-specific test set.
      - [ ] Implement Hit@k metrics for various k values (1, 3, 5, 10)
      - [ ] Calculate Mean Reciprocal Rank (MRR) across test queries
      - [ ] Measure normalized Discounted Cumulative Gain (nDCG)
      - [ ] Compare performance against BM25, FAISS (cosine), Chroma/HNSW, and hybrids
    - [ ] Develop specific tests for RAGatouille's late-interaction capabilities.
    - [ ] Create visualization tools for retrieval quality assessment.
    - [ ] Implement A/B testing framework to compare retrieval methods.
  - [ ] **Edge Cases & Robustness**:
    - [ ] Develop fallback strategies for when RAGatouille fails or underperforms.
    - [ ] Create handling for very long documents and queries.
    - [ ] Implement error recovery for model loading and prediction issues.
    - [ ] Design graceful degradation for resource-constrained environments.

- **Outcome**: A high-performance retrieval system leveraging RAGatouille's advanced late-interaction ColBERT approach, providing superior semantic search while integrating seamlessly with the existing architecture. The implementation will include hybrid approaches, robust fallbacks, and comprehensive testing to ensure reliable performance across various document types.

## Comparison Implementation Details & Strategies

1.  **Document Comparison Logic**:

    - **MVP (Phase 6) - COMPLETED**:
      - Parse structured text (Markdown/JSON) into sections/paragraphs.
      - Align sections (heading matching, sequence).
      - Generate embeddings for aligned pairs.
      - Use cosine similarity with thresholds (e.g., >0.95 = Similar, <0.7 = Different, intermediate = Check Needed).
    - **Enhancement (Phase 7) - PROTOTYPING INITIATED & REFINED**:
      - Feed pairs flagged by embeddings (e.g., `DIFFERENT`, `MODIFIED`) to an LLM (`LLMComparer` with `OllamaClient`). (Short content skip added)
      - Prompt the LLM to classify the difference and explain (`DEFAULT_PROMPT_TEMPLATE` in `LLMComparer`, `LLMAnalysisResult` model defined).
    - **Future Enhancement (Post-Phase 7):**
      - _If needed:_ Use ColBERT for token-level similarity assessment on ambiguous segments.

2.  **Handling OCR Noise During Comparison**:

    - **Primary Strategy:** Rely on the LLM Cleaning step (Phase 3) to produce clean text _before_ comparison. (Enabled in E2E script via `OCRPipelineConfig`).
    - **Secondary Strategy (Phase 7):** Instruct the comparison LLM (via prompts) to be robust to minor variations typical of OCR noise when assessing semantic meaning. Embeddings are often inherently robust to minor noise.

3.  **Handling Different Layouts**:

    - **Core Strategy:** Compare the _structured text output_ (Markdown/JSON) generated by the OCR pipeline. This largely abstracts away the original visual layout.
    - **Alignment Robustness:** The section alignment logic needs to handle cases where sections might be reordered or split/merged. Start simple (sequence/headings), enhance with algorithms like Needleman-Wunsch if needed, potentially using embedding similarity to guide alignment.

4.  **Evaluating Semantic Similarity & Detecting Rewrites**:

    - **Embeddings (Phase 6) - COMPLETED**: Cosine similarity provides a good first pass for overall semantic similarity.
    - **LLMs (Phase 7) - PROTOTYPING INITIATED & REFINED**: Provide explicit prompts:
      - Current `DEFAULT_PROMPT_TEMPLATE` in `LLMComparer` aims to achieve this.
    - **Future Enhancement (Post-Phase 7):**
      - _If needed:_ ColBERT token-level interactions provide fine-grained comparison, potentially effective for detecting rewrites and semantic-preserving edits.
    - **Hybrid - IMPLEMENTED & REFINED**: Use embeddings for speed, trigger LLM analysis for ambiguous/different cases (with short content skip).

5.  **Presenting the Comparison**:
    - **MVP (Markdown - Phase 6) - COMPLETED**:
      - Use simple block annotations: `[SIMILAR]`, `[DIFFERENT]`, `[NEW]`, `[DELETED]`.
      - Show content for `DIFFERENT` sections using `---` (Doc A) and `+++` (Doc B).
    - **Enhanced (Markdown/HTML/Text - Phase 7) - IN PROGRESS**:
      - [x] More descriptive Markdown annotations based on LLM analysis: `[MODIFIED MEANING (LLM)]`, `[REWRITTEN - SIMILAR MEANING (LLM)]`, etc. (Implemented in `DiffFormatter` for MD, HTML, Text)
      - [x] Include LLM explanation in the diff output. (Implemented in `DiffFormatter`)
      - [ ] Include standard text diffs (e.g., from `difflib`) within sections marked as different. (Already present for detailed Markdown)
      - _Future:_ Integrate ColBERT token-level highlighting if implemented.
      - [ ] Generate HTML for interactive side-by-side views with color highlighting for different change types. (Basic HTML structure exists; enhancements for specific LLM categories can be added)

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
      - (Optionally) Interfaces with **Comparison LLM Service/Library** (Prototyped with `LLMComparer` and `OllamaClient`).
      - _Future Investigation:_ Integration with **ColBERT**.
      - Calculates similarity and/or performs LLM analysis on aligned pairs.
    - **`Diff Formatter`**: Generates the final report (Markdown/HTML).
    - **Note on Architecture**: Consider refactoring this service to follow the modular pattern established in the RAG pipeline, with clear separation of responsibilities and well-defined interfaces between components. (This refactoring has been substantially completed by centralizing domain models and defining component protocols.)

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

## User Interface Implementation with Open WebUI

- **Goal**: Integrate Open WebUI as a user-friendly frontend interface for the RAG system, providing a polished, ready-made solution for interacting with the backend.
- **Tasks**:

  - [ ] **Initial Setup & Integration**:
    - [ ] Deploy Open WebUI alongside the existing FastAPI backend.
    - [ ] Configure Open WebUI to connect to the RAG API endpoints.
    - [ ] Test basic query functionality through the UI.
  - [ ] **Adapter Development**:
    - [ ] Create adapter endpoints in the FastAPI backend if needed to match Open WebUI's expected API format.
    - [ ] Implement conversion between RAG response format and Open WebUI's expected response structure.
    - [ ] Add support for conversation history persistence.
  - [ ] **Custom Enhancements**:
    - [ ] Configure Open WebUI to display document citations and sources retrieved by the RAG system.
    - [ ] Customize the UI theme and branding if needed.
    - [ ] Implement user authentication and access control if required.
  - [ ] **Feature Integration**:
    - [ ] Enable support for document comparison visualization through the UI.
    - [ ] Add document upload functionality for processing new documents.
    - [ ] Implement visualization of OCR results and document structuring.
  - [ ] **Testing & Documentation**:
    - [ ] Develop comprehensive end-to-end tests for the integrated system.
    - [ ] Create documentation for users explaining the UI features.
    - [ ] Document the integration process for developers.
  - [ ] **Deployment Configuration**:
    - [ ] Update Docker Compose and Kubernetes configurations to include Open WebUI.
    - [ ] Configure proper networking between the frontend and backend services.
    - [ ] Set up appropriate environment variables for connecting components.

- **Outcome**: A fully functional, user-friendly web interface for interacting with the RAG system using Open WebUI's polished components, while maintaining the flexibility to switch to a different frontend solution in the future if needed.

## Proposed Frontend Architecture

The implementation will follow these architectural principles:

1. **Loose Coupling**: The Open WebUI frontend will communicate with the backend exclusively through the API, ensuring minimal dependencies between systems.

2. **API Contract**: Establish a clear API contract that defines the interface between frontend and backend, allowing for potential frontend replacement in the future.

3. **Backend Storage**: Store user data like conversations in the backend database rather than in Open WebUI's storage, avoiding data migration challenges if frontend changes are needed later.

4. **Independent Deployment**: Configure the system to allow independent deployment and scaling of frontend and backend components.

This approach provides immediate access to a polished UI while preserving the flexibility to develop a custom frontend or integrate a different solution as requirements evolve.
