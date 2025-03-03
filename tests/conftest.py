"""
Test configuration for pytest.

This file sets up the Python path so tests can import modules from the src directory.
"""

import importlib
import os
import sys

# Add the src directory to the Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import src.llm_rag to ensure it's loaded
src_llm_rag = importlib.import_module("src.llm_rag")

# Create a mapping for llm_rag and its submodules
sys.modules["llm_rag"] = src_llm_rag

# Map each submodule
submodules = ["api", "models", "vectorstore", "document_processing", "rag"]
for submodule in submodules:
    try:
        # Try to import the submodule from src.llm_rag
        module = importlib.import_module(f"src.llm_rag.{submodule}")
        # Create an alias for the module
        sys.modules[f"llm_rag.{submodule}"] = module
    except ImportError:
        # If the submodule doesn't exist, create an empty module
        pass
