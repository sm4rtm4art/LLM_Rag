"""Unit tests for the document loaders entry point module."""

from unittest.mock import patch


def test_successful_imports():
    """Test that the module imports successfully."""
    # Import the main entry point classes and functions
    from llm_rag.document_processing.loaders import (
        DirectoryLoader,
        DocumentLoader,
        FileLoader,
        load_document,
        load_documents_from_directory,
    )

    # Verify the imports were successful
    assert DocumentLoader is not None
    assert FileLoader is not None
    assert DirectoryLoader is not None
    assert load_document is not None
    assert load_documents_from_directory is not None


def test_stub_classes_used_when_imports_fail():
    """Test that stub classes are used when imports fail."""
    with patch("llm_rag.document_processing.loaders._MODULAR_IMPORT_SUCCESS", False):
        # Import should still succeed even with failed submodule imports
        from llm_rag.document_processing.loaders import FileLoader, load_document

        # Create a test file loader
        loader = FileLoader("test.txt")
        result = loader.load()

        # Stub methods should return empty lists
        assert isinstance(result, list)
        assert len(result) == 0

        # Test load_document returns None as it's a stub
        assert load_document("nonexistent.txt") is None


def test_backward_compatibility_with_loader_api():
    """Test backward compatibility with the loader_api module."""
    # Import from both modules
    from llm_rag.document_processing.loader_api import DocumentLoader as OldDocumentLoader
    from llm_rag.document_processing.loaders import DocumentLoader as NewDocumentLoader

    # They should be the same class
    assert NewDocumentLoader is OldDocumentLoader

    # Import functions and classes
    from llm_rag.document_processing.loader_api import JSONLoader as OldJSONLoader
    from llm_rag.document_processing.loader_api import PDFLoader as OldPDFLoader
    from llm_rag.document_processing.loader_api import load_document as old_load_document
    from llm_rag.document_processing.loaders import JSONLoader, PDFLoader
    from llm_rag.document_processing.loaders import load_document as new_load_document

    # They should be the same
    assert JSONLoader is OldJSONLoader
    assert PDFLoader is OldPDFLoader
    assert new_load_document is old_load_document
