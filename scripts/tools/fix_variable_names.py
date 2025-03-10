#!/usr/bin/env python3
# flake8: noqa: E501
"""Script to fix variable naming inconsistencies in the codebase."""

import logging
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fix_pipeline_py():
    """Fix variable naming inconsistencies in pipeline.py."""
    file_path = "src/llm_rag/rag/pipeline.py"

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    with open(file_path, "r") as f:
        content = f.read()

    # Define replacements
    replacements = [
        (r"\braw_docs\b", "documents"),
        (r"\bdocs\b", "documents"),
        (r"\bdoc_count\b", "document_count"),
        (r"\bdoc_index\b", "document_index"),
        (r"\bformatted_docs\b", "formatted_documents"),
        (r"\bprocessed_doc\b", "processed_document"),
        (r"\bconfidence\b", "config"),
        (r"\bconfidence_warning\b", "config_warning"),
        (r"\bvector_store\b", "vectorstore"),
    ]

    # Apply replacements
    original_content = content
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # Write back if changes were made
    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        logger.info(f"Fixed variable names in {file_path}. Characters changed: {len(content) - len(original_content)}")
    else:
        logger.info(f"No changes needed in {file_path}")


def fix_anti_hallucination_py():
    """Fix variable naming inconsistencies in anti_hallucination.py."""
    file_path = "src/llm_rag/rag/anti_hallucination.py"

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    with open(file_path, "r") as f:
        content = f.read()

    # Define replacements
    replacements = [
        (r"\bconfidence_level\b", "config_level"),
        (r"\bembedding_similarity\b", "embeddings_similarity"),
        (r"\bembed_sim\b", "embeddings_sim"),
        (r"\buse_embeddings\b", "use_embeddings_verification"),
        (r"\bstopwords_path\b", "stopwords_directory"),
    ]

    # Apply replacements
    original_content = content
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # Write back if changes were made
    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        logger.info(f"Fixed variable names in {file_path}. Characters changed: {len(content) - len(original_content)}")
    else:
        logger.info(f"No changes needed in {file_path}")


def main():
    """Run the variable name fixing script."""
    logger.info("Starting to fix variable names...")
    fix_pipeline_py()
    fix_anti_hallucination_py()
    logger.info("Variable name fixes completed.")


if __name__ == "__main__":
    main()
