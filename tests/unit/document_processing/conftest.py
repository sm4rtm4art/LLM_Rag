#!/usr/bin/env python3
"""Test configuration for document processing tests."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_file_exists():
    """Mock Path.exists to return True."""
    with patch("pathlib.Path.exists", return_value=True) as mock:
        yield mock


@pytest.fixture
def mock_is_dir():
    """Mock Path.is_dir to return True."""
    with patch("pathlib.Path.is_dir", return_value=True) as mock:
        yield mock


@pytest.fixture
def mock_file_open():
    """Mock built-in open function."""
    with patch("builtins.open") as mock:
        yield mock


@pytest.fixture
def mock_registry():
    """Mock the global loader registry."""
    with patch("llm_rag.document_processing.loaders.factory.registry") as mock:
        yield mock


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "refactoring: mark test as being refactored")


def pytest_collection_modifyitems(config, items):
    """Skip tests marked as refactoring if SKIP_REFACTORING_TESTS is set."""
    skip_refactoring = pytest.mark.skip(reason="Test is being refactored")

    # Skip tests being refactored if env var is set (for CI/CD)
    if os.environ.get("SKIP_REFACTORING_TESTS", "").lower() in ("1", "true", "yes"):
        for item in items:
            if "refactoring" in item.keywords:
                item.add_marker(skip_refactoring)
