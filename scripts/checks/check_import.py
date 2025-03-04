#!/usr/bin/env python3
"""Check if required modules are available for import."""

import importlib.util
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_module_available(module_name):
    """Check if a module is available for import.

    Args:
        module_name: Name of the module to check

    Returns:
        bool: True if the module is available, False otherwise

    """
    spec = importlib.util.find_spec(module_name)
    return spec is not None


def check_imports():
    """Check if required modules are available for import.

    Returns:
        bool: True if all required modules are available, False otherwise

    """
    logger.info(f"Python version: {sys.version}")
    logger.info(f"sys.path: {sys.path}")

    # List of modules to check
    modules_to_check = [
        "llm_rag",
        "langchain",
        "langchain_community",
        "transformers",
        "sentence_transformers",
        "chromadb",
        "torch",
        "pypdf",
        "llama_cpp",
    ]

    all_available = True

    for module_name in modules_to_check:
        if check_module_available(module_name):
            logger.info(f"{module_name} module is available")
        else:
            logger.warning(f"{module_name} module is NOT available")
            all_available = False

    return all_available


def main():
    """Run the import check."""
    success = check_imports()
    if success:
        logger.info("All required modules are available")
    else:
        logger.warning("Some required modules are missing")


if __name__ == "__main__":
    main()
