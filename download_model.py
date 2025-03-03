#!/usr/bin/env python
"""Download script for LLM models.

This script helps download LLM models from HuggingFace for use with the RAG demo.
It supports downloading models from TheBloke's repository by default.
"""

import argparse
import os
import sys

from huggingface_hub import hf_hub_download


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser.

    Returns
    -------
        An argument parser for the download script.

    """
    parser = argparse.ArgumentParser(description="Download LLM models for RAG demo")

    parser.add_argument(
        "--model",
        type=str,
        default="llama-2-7b-chat.Q4_K_M.gguf",
        help="Model filename to download",
    )

    parser.add_argument(
        "--repo",
        type=str,
        default="TheBloke/Llama-2-7B-Chat-GGUF",
        help="HuggingFace repository containing the model",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save the model",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if the model already exists",
    )

    return parser


def download_model(repo_id: str, filename: str, output_dir: str, force: bool = False) -> bool:
    """Download a model from HuggingFace Hub.

    Args:
    ----
        repo_id: HuggingFace repository ID (e.g., 'TheBloke/Llama-2-7B-Chat-GGUF')
        filename: Model filename to download (e.g., 'llama-2-7b-chat.Q4_K_M.gguf')
        output_dir: Directory to save the model
        force: Whether to force download even if file exists

    Returns:
    -------
        True if download succeeded, False otherwise

    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    # Check if file already exists
    if os.path.exists(output_path) and not force:
        print(f"Model {filename} already exists at {output_path}")
        print("Use --force to download anyway")
        return True

    print(f"Downloading {filename} from {repo_id}")
    print("This may take a while depending on the model size...")

    try:
        # Download the model using huggingface_hub
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )

        print(f"Successfully downloaded {filename} to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False


def main() -> None:
    """Main entry point for the download script."""
    # Parse command-line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Download the model
    success = download_model(
        repo_id=args.repo,
        filename=args.model,
        output_dir=args.output_dir,
        force=args.force,
    )

    if success:
        print("\nYou can now use the model with the RAG demo:")
        print(f"python demo_llm_rag.py --model-path {os.path.join(args.output_dir, args.model)}")
    else:
        print("\nModel download failed. Please check your internet connection or repository/filename.")
        sys.exit(1)


if __name__ == "__main__":
    main()
