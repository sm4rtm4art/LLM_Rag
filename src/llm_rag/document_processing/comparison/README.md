# Document Comparison Module

## Purpose

This module is responsible for comparing two documents to identify and report differences and similarities between them. It is designed to work with structured text output (primarily Markdown, with extensibility for other formats) that may originate from an OCR pipeline or other document processing stages.

The comparison is performed section by section, and the module aims to provide a human-readable report detailing the changes. A key feature is the optional integration of a Large Language Model (LLM) for nuanced semantic analysis of differing sections, providing deeper insights beyond simple text-based diffs.

## High-Level Workflow

The comparison process generally follows these steps:

1.  **Parsing**: Input documents (provided as content strings or file paths) are parsed into a list of logical `Section` objects. The parser can handle different input formats (e.g., Markdown, pre-chunked text).
2.  **Alignment**: The `Section` lists from the two documents are aligned to identify corresponding sections. Various strategies can be used for alignment, including heading matches, content similarity, and sequence analysis.
3.  **Embedding-Based Comparison**: Aligned section pairs are first compared using semantic embeddings. A similarity score is calculated, and based on configurable thresholds, the pair is classified as `SIMILAR`, `MODIFIED`, or `DIFFERENT`. Sections unique to one document are marked as `NEW` or `DELETED`.
4.  **LLM-Enhanced Analysis (Optional)**:
    - For sections classified as `MODIFIED` or `DIFFERENT` by the embedding comparison (and if the content is substantial enough), an optional, more detailed analysis can be performed using an LLM (e.g., via an Ollama client).
    - The LLM is prompted to categorize the nature of the difference (e.g., `LEGAL_EFFECT_CHANGE`, `SEMANTIC_REWRITE`, `DIFFERENT_CONCEPTS`) and provide a textual explanation and confidence score.
    - This `LLMAnalysisResult` is stored alongside the primary comparison result.
5.  **Formatting**: The list of `SectionComparison` objects (each potentially containing an `LLMAnalysisResult`) is formatted into a human-readable report (e.g., Markdown, HTML, Text). The report highlights the type of change for each section and includes the detailed LLM analysis if performed.

## Key Components

- **`DocumentParser` (`document_parser.py`)**: Parses input document content into `Section` objects.
- **`SectionAligner` (`alignment.py`)**: Aligns sections between two parsed documents.
- **`EmbeddingComparisonEngine` (`comparison_engine.py`)**: Performs embedding-based similarity scoring and classification. Orchestrates optional LLM analysis by invoking `LLMComparer`.
- **`LLMComparer` (`llm_comparer.py`)**: Interfaces with an LLM client to get detailed semantic analysis for pairs of text sections. Parses the LLM's structured JSON output.
- **`DiffFormatter` (`diff_formatter.py`)**: Generates the final human-readable diff report in various formats, incorporating LLM analysis details.
- **`ComparisonPipeline` (`pipeline.py`)**: Orchestrates the entire workflow from input documents to final diff report, managing configurations for all components.
- **Domain Models (`domain_models.py`)**: Defines Pydantic and dataclass models for configuration, sections, alignment pairs, comparison results (including `LLMAnalysisResult`), etc.
- **LLM Clients (e.g., `src/llm_rag/llm_clients/ollama_client.py`)**: Provides the interface to the actual LLM services.

## Basic Usage / Integration

The `ComparisonPipeline` is the main entry point for using this module. It is initialized with a `ComparisonPipelineConfig` object that specifies settings for all underlying components, including parser options, alignment strategy, similarity thresholds, formatter preferences, and LLM analysis parameters (e.g., whether to enable it, which LLM model to use).

```python
# Example (conceptual)
from llm_rag.document_processing.comparison import ComparisonPipeline, ComparisonPipelineConfig
from llm_rag.document_processing.comparison.domain_models import DocumentFormat

# Configure the pipeline (enable LLM analysis)
config = ComparisonPipelineConfig()
config.llm_comparer_pipeline_config.enable_llm_analysis = True
# ... other configurations for parser, aligner, embedding model, LLM model, etc.

pipeline = ComparisonPipeline(config=config)

async def run_comparison():
    diff_report = await pipeline.compare_documents(
        source_document="path/to/doc1.md",
        target_document="path/to/doc2.md",
        source_format=DocumentFormat.MARKDOWN,
        target_format=DocumentFormat.MARKDOWN,
        title="Comparison of Document Alpha vs. Beta"
    )
    print(diff_report)

# asyncio.run(run_comparison())
```

The module is designed to be integrated into larger document processing workflows, potentially taking its input from an OCR pipeline that outputs structured Markdown.

## LLM Enhancement for Detailed Analysis

The integration of an LLM allows the comparison module to go beyond simple similarity scores. When enabled:

- Sections flagged as `MODIFIED` or `DIFFERENT` are sent to an LLM.
- The LLM provides a specific `comparison_category` tailored for analyzing changes, particularly in legal or formal documents (e.g., `SEMANTIC_REWRITE`, `LEGAL_EFFECT_CHANGE`).
- An `explanation` and `confidence` score from the LLM are also captured.
- These details are included in the final diff report, offering much richer insights into how sections differ, rather than just that they differ.
- A check is in place to skip LLM analysis for very short or empty sections to improve efficiency, in which case a category of `NO_MEANINGFUL_CONTENT` is assigned.
