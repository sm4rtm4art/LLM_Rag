"""
Test configuration for pytest.

This file sets up the Python path so tests can import modules from the src directory.
"""

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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

# Root directory of the project
TEST_ROOT = Path(__file__).parent
PROJECT_ROOT = TEST_ROOT.parent
TEST_DATA_DIR = TEST_ROOT / "test_data"

# Only mock these if they're causing issues with Cursor
if "CURSOR_SESSION" in os.environ:
    # Create mocks for problematic modules in Cursor only
    MOCK_MODULES = ["torch", "sentence_transformers", "chromadb"]
    for mod_name in MOCK_MODULES:
        sys.modules[mod_name] = MagicMock()


# Fixtures for accessing test data
@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def sample_text_file():
    """Return the path to a sample text file for testing."""
    return TEST_DATA_DIR / "sample.txt"


@pytest.fixture
def sample_documents():
    """Return a list of sample documents for testing."""
    return [
        {"content": "This is test document 1.", "metadata": {"source": "test1.txt"}},
        {"content": "This is test document 2.", "metadata": {"source": "test2.txt"}},
    ]


# Mock fixtures for common components
@pytest.fixture
def mock_vectorstore():
    """Create a mock vector store for testing."""

    class MockVectorStore:
        def __init__(self, *args, **kwargs):
            self.documents = []

        def add_documents(self, documents):
            self.documents.extend(documents)
            return [f"doc_id_{i}" for i in range(len(documents))]

        def similarity_search(self, query, k=4):
            return (
                self.documents[: min(k, len(self.documents))]
                if self.documents
                else [{"content": f"Result for {query}", "metadata": {"source": "mock"}}] * k
            )

    return MockVectorStore()


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.predict.return_value = "This is a mock response about the query."
    return mock


# Now the rest of your conftest code can run without those dependencies
