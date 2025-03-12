#!/usr/bin/env python3
"""Test configuration for document processing tests."""

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
