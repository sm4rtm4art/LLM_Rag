"""Examples of language preservation and translation in the OCR pipeline.

This module provides examples of using the OCR pipeline with language awareness,
showing how to preserve the original language of a document or translate it to another language.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

from llm_rag.document_processing.ocr.llm_processor import LLMCleanerConfig
from llm_rag.document_processing.ocr.pipeline import OCRPipeline, OCRPipelineConfig
from llm_rag.utils.logging import get_logger

logger = get_logger(__name__)


def process_document_with_language_handling(
    pdf_path: str,
    preserve_language: bool = True,
    target_language: Optional[str] = None,
    output_dir: Optional[str] = None,
    language_models: Optional[Dict[str, str]] = None,
) -> str:
    """Process a document with language-aware OCR.

    Args:
        pdf_path: Path to the PDF file to process
        preserve_language: Whether to preserve the original language (default: True)
        target_language: Target language for translation (default: None)
        output_dir: Directory to save the results (default: None)
        language_models: Dictionary mapping language codes to model names (default: None)

    Returns:
        Processed text

    """
    # Create a configuration that handles language appropriately
    cleaner_config = LLMCleanerConfig(
        model_name="gemma-2b",  # Fast, lightweight model
        model_backend="ollama",  # Using local Ollama (requires installation)
        preserve_language=preserve_language,
        translate_to_language=target_language,
        language_models=language_models or {},
        detect_language=True,
    )

    # Create OCR pipeline configuration
    pipeline_config = OCRPipelineConfig(
        llm_cleaner_config=cleaner_config,
        use_llm_cleaner=True,  # Enable LLM cleaning
        output_format="markdown",  # Output as markdown for readability
    )

    # Create and run the pipeline
    pipeline = OCRPipeline(config=pipeline_config)

    logger.info(f"Processing document: {pdf_path}")
    logger.info(f"Language preservation: {preserve_language}")
    if target_language:
        logger.info(f"Target language for translation: {target_language}")

    # Process the document
    result = pipeline.process_pdf(pdf_path)

    # Save results if output_dir is specified
    if output_dir:
        save_path = Path(output_dir) / f"{Path(pdf_path).stem}_processed.md"
        os.makedirs(output_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(result)
        logger.info(f"Saved processed text to: {save_path}")

    return result


def main():
    """Run the language-aware OCR example from the command line."""
    parser = argparse.ArgumentParser(description="Language-aware OCR processing example")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument(
        "--preserve-language",
        action="store_true",
        default=True,
        help="Preserve the original language (default: True)",
    )
    parser.add_argument(
        "--translate-to",
        help="Target language for translation (e.g., 'en', 'de', 'fr')",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save the processed text",
        default="./processed_docs",
    )
    args = parser.parse_args()

    # Example language-specific model mapping
    # These are just examples - replace with actual models you have access to
    language_models = {
        "de": "german-llm-model",  # Example for German
        "fr": "french-llm-model",  # Example for French
        # Add more language-specific models as needed
    }

    # Process the document
    process_document_with_language_handling(
        pdf_path=args.pdf_path,
        preserve_language=args.preserve_language,
        target_language=args.translate_to,
        output_dir=args.output_dir,
        language_models=language_models,
    )


if __name__ == "__main__":
    main()
