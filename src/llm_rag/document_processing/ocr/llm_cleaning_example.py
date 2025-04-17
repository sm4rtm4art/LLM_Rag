"""Example demonstrating the OCR pipeline with LLM text cleaning."""

import argparse
import os
import sys
from pathlib import Path

from llm_rag.document_processing.ocr.pipeline import OCRPipeline, OCRPipelineConfig
from llm_rag.utils.logging import setup_logging


def main():
    """Run the OCR pipeline with LLM cleaning example."""
    parser = argparse.ArgumentParser(description="OCR with LLM text cleaning")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to process")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save output files (default: output/ocr/)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "txt"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--model", type=str, default="gemma-2b", help="LLM model to use for cleaning (default: gemma-2b)"
    )
    parser.add_argument(
        "--model-backend",
        type=str,
        choices=["ollama", "huggingface", "llama_cpp"],
        default="ollama",
        help="Backend to use for the model (default: ollama)",
    )
    parser.add_argument("--languages", type=str, default="eng", help="OCR language(s), comma-separated (default: eng)")
    parser.add_argument(
        "--min-error-rate",
        type=float,
        default=0.05,
        help="Only apply LLM cleaning when estimated error rate is above this threshold",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Only apply LLM cleaning when OCR confidence is below this threshold",
    )
    parser.add_argument("--no-llm-cleaning", action="store_true", help="Disable LLM cleaning (for comparison)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = "debug" if args.verbose else "info"
    setup_logging(level=log_level)

    # Parse languages
    languages = args.languages.split(",")
    if len(languages) == 1:
        languages = languages[0]

    # Create OCR pipeline configuration
    config = OCRPipelineConfig(
        # Enable preprocessing to improve OCR quality
        preprocessing_enabled=True,
        deskew_enabled=True,
        contrast_adjust=1.2,
        # OCR settings
        languages=languages,
        # Output format settings
        output_format=args.format,
        detect_headings=True,
        detect_lists=True,
        detect_tables=True,
        # LLM cleaning settings
        llm_cleaning_enabled=not args.no_llm_cleaning,
        llm_model_name=args.model,
        llm_model_backend=args.model_backend,
        llm_confidence_threshold=args.confidence_threshold,
        llm_min_error_rate=args.min_error_rate,
        llm_preserve_layout=True,
    )

    # Initialize the pipeline
    pipeline = OCRPipeline(config=config)

    # Process the PDF and save the results
    try:
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            print(f"Error: PDF file not found: {pdf_path}")
            return 1

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Create a structured output directory based on processing type
            base_output_dir = Path("output/ocr")
            # Use subdirectory based on whether LLM cleaning is enabled
            subdir = "with_llm" if not args.no_llm_cleaning else "without_llm"
            # Use format as another subdirectory level
            format_dir = args.format
            # Combine to create final output path
            output_dir = base_output_dir / subdir / format_dir

        # Create the output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing PDF: {pdf_path}")
        print(f"LLM cleaning {'enabled' if not args.no_llm_cleaning else 'disabled'}")
        print(f"Output directory: {output_dir}")

        if args.no_llm_cleaning:
            print("Running without LLM cleaning...")
        else:
            print(f"Using LLM model: {args.model} with backend: {args.model_backend}")
            print(f"Error rate threshold: {args.min_error_rate}")
            print(f"Confidence threshold: {args.confidence_threshold}")

        output_path = pipeline.process_and_save(pdf_path, output_dir, args.format)

        print(f"Processing complete! Output saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def process_directory(input_dir, output_dir=None, **kwargs):
    """Process all PDF files in a directory.

    Args:
        input_dir: Directory containing PDF files to process
        output_dir: Optional output directory (uses default structure if None)
        **kwargs: Additional arguments to pass to OCRPipelineConfig

    Returns:
        Number of files successfully processed

    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory not found: {input_path}")
        return 0

    # Find all PDF files
    pdf_files = list(input_path.glob("**/*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_path}")
        return 0

    print(f"Found {len(pdf_files)} PDF files to process")
    successful = 0

    for pdf_file in pdf_files:
        try:
            # Construct command-line arguments
            sys_args = [str(pdf_file)]

            if output_dir:
                sys_args.extend(["--output-dir", str(output_dir)])

            # Add any other kwargs as command-line arguments
            for key, value in kwargs.items():
                if value is True:  # For boolean flags
                    sys_args.append(f"--{key.replace('_', '-')}")
                elif value is not False:  # Skip if False
                    sys_args.append(f"--{key.replace('_', '-')}")
                    sys_args.append(str(value))

            # Save original sys.argv
            original_argv = sys.argv

            try:
                # Replace sys.argv with our constructed arguments
                sys.argv = [sys.argv[0]] + sys_args

                # Run main with these arguments
                result = main()

                if result == 0:
                    successful += 1

            finally:
                # Restore original sys.argv
                sys.argv = original_argv

        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

    print(f"Successfully processed {successful} out of {len(pdf_files)} PDF files")
    return successful


if __name__ == "__main__":
    # Check if we're processing a directory
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        # First argument is a directory
        input_dir = sys.argv[1]

        # Remove the directory argument to prevent confusing the argument parser
        sys.argv.pop(1)

        # Process all PDFs in the directory
        process_directory(input_dir)
    else:
        # Process a single PDF file
        sys.exit(main())
