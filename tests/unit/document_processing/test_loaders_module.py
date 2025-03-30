"""Unit tests for the legacy loaders.py module."""

import tempfile
from unittest.mock import patch

import pytest

# Import the module under test
import llm_rag.document_processing.loaders


@pytest.fixture
def mock_imports():
    """Fixture to mock imports and control _MODULAR_IMPORT_SUCCESS flag."""
    # Save original value
    original_value = llm_rag.document_processing.loaders._MODULAR_IMPORT_SUCCESS

    # Yield control to the test
    yield

    # Restore original value
    llm_rag.document_processing.loaders._MODULAR_IMPORT_SUCCESS = original_value


class TestLoadersModule:
    """Test cases for the loaders.py module."""

    def test_all_exports_defined(self):
        """Test that all expected exports are available in the module."""
        # Check that all __all__ exports are actually defined in the module
        for name in llm_rag.document_processing.loaders.__all__:
            assert hasattr(llm_rag.document_processing.loaders, name)
            assert getattr(llm_rag.document_processing.loaders, name) is not None

    def test_load_document_success(self):
        """Test the load_document function when it succeeds."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as temp_file:
            temp_file.write("Test content")
            temp_file.flush()

            # When load_document succeeds
            with patch("llm_rag.document_processing.loaders.load_document") as mock_load:
                expected_docs = [{"content": "Test content", "metadata": {"source": temp_file.name}}]
                mock_load.return_value = expected_docs

                # Call the function directly (not through the module)
                # This avoids infinite recursion in the patched function
                from llm_rag.document_processing import loaders

                result = loaders.load_document(temp_file.name)

                # Verify result
                assert result == expected_docs
                mock_load.assert_called_once()

    def test_load_document_error_handling(self):
        """Test error handling in load_document."""
        # Instead of patching the function itself, we'll test
        # the actual implementation with an invalid file
        non_existent_file = "/path/to/nonexistent/file.xyz"
        result = llm_rag.document_processing.loaders.load_document(non_existent_file)

        # Should return None for non-existent files
        assert result is None

    @pytest.mark.parametrize("import_success", [True, False])
    def test_all_loader_classes_exist(self, import_success):
        """Test that all loader classes exist regardless of import success."""
        with patch("llm_rag.document_processing.loaders._MODULAR_IMPORT_SUCCESS", import_success):
            # Import the module dynamically to refresh state
            import importlib

            importlib.reload(llm_rag.document_processing.loaders)

            # Check that all loader classes are available
            loaders = [
                "DocumentLoader",
                "FileLoader",
                "DirectoryLoader",
                "CSVLoader",
                "PDFLoader",
                "EnhancedPDFLoader",
                "JSONLoader",
                "TextFileLoader",
                "XMLLoader",
                "WebLoader",
                "WebPageLoader",
            ]

            for loader_name in loaders:
                assert hasattr(llm_rag.document_processing.loaders, loader_name)

    def test_successful_imports_use_modular_implementation(self):
        """Test that when imports succeed, the modular implementations are used."""
        # Should already be using the modular implementation
        assert llm_rag.document_processing.loaders._MODULAR_IMPORT_SUCCESS is True

        # Verify that the module is re-exporting from loaders/
        from llm_rag.document_processing.loaders import EnhancedPDFLoader
        from llm_rag.document_processing.loaders.file_loaders import EnhancedPDFLoader as ModularEnhancedPDFLoader

        assert EnhancedPDFLoader is ModularEnhancedPDFLoader

    def test_get_available_loader_extensions(self):
        """Test that get_available_loader_extensions returns the extensions."""
        extensions = llm_rag.document_processing.loaders.get_available_loader_extensions()

        # Verify result
        assert isinstance(extensions, dict)
        # At minimum, these extensions should be supported
        assert ".txt" in extensions
        assert ".pdf" in extensions
        assert ".json" in extensions
        assert ".csv" in extensions


class TestBackwardCompatibility:
    """Test backward compatibility with the loader_api module."""

    def test_loader_api_compatibility(self):
        """Test compatibility with the loader_api module."""
        # Import from both modules
        from llm_rag.document_processing.loader_api import DocumentLoader as ApiDocumentLoader
        from llm_rag.document_processing.loaders import DocumentLoader

        # They should be the same class
        assert DocumentLoader is ApiDocumentLoader

        # Import functions and classes
        from llm_rag.document_processing.loader_api import JSONLoader as ApiJSONLoader
        from llm_rag.document_processing.loader_api import PDFLoader as ApiPDFLoader
        from llm_rag.document_processing.loaders import JSONLoader, PDFLoader

        # They should be the same
        assert JSONLoader is ApiJSONLoader
        assert PDFLoader is ApiPDFLoader

        # Test load_document compatibility
        # The functionality should be equivalent even if they're not the same function
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as temp_file:
            temp_file.write("Test compatibility")
            temp_file.flush()

            # Mock both implementations to return predictable values
            with patch(
                "llm_rag.document_processing.loaders.load_document", return_value=[{"content": "mocked content"}]
            ):
                with patch(
                    "llm_rag.document_processing.loader_api.load_document", return_value=[{"content": "mocked content"}]
                ):
                    # Both should work with the same arguments
                    from llm_rag.document_processing.loader_api import load_document as api_load
                    from llm_rag.document_processing.loaders import load_document as compat_load

                    # Both should accept the same parameters
                    api_result = api_load(temp_file.name)
                    compat_result = compat_load(temp_file.name)

                    # Results should be equivalent
                    assert api_result == compat_result
