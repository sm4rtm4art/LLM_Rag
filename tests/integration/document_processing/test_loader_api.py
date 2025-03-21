"""Integration tests for loader_api.py entry point.

These tests verify that the loader_api module correctly re-exports loader components
and maintains backward compatibility with various import patterns.
"""

import tempfile
from pathlib import Path


class TestLoaderAPIIntegration:
    """Integration tests for the loader_api module entry point."""

    def setup_method(self):
        """Set up test environment."""
        # Create temporary test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create a text file
        self.txt_file = self.test_dir / "test.txt"
        self.txt_file.write_text("This is a test document.")

        # Create a JSON file
        self.json_file = self.test_dir / "test.json"
        self.json_file.write_text('{"title": "Test", "content": "JSON test content"}')

    def teardown_method(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_import_patterns(self):
        """Test that various loader import patterns work correctly."""
        # Direct imports from loader_api
        from llm_rag.document_processing.loader_api import DocumentLoader, JSONLoader, TextFileLoader

        # Verify classes were imported correctly
        assert TextFileLoader is not None
        assert JSONLoader is not None
        assert issubclass(TextFileLoader, DocumentLoader)
        assert issubclass(JSONLoader, DocumentLoader)

    def test_loader_functionality(self):
        """Test the loader functionality."""
        # Import a loader class directly
        from llm_rag.document_processing.loader_api import TextFileLoader

        # Create a loader and load the file
        loader = TextFileLoader(file_path=self.txt_file)
        result = loader.load()

        # Verify results
        assert result is not None
        assert len(result) == 1
        assert result[0]["content"] == "This is a test document."
        assert result[0]["metadata"]["source"] == str(self.txt_file)
        assert result[0]["metadata"]["filename"] == "test.txt"

    def test_directory_loading(self):
        """Test loading from a directory through the loader_api entry point."""
        # Import the directory loading function
        from llm_rag.document_processing.loader_api import DirectoryLoader, load_documents_from_directory

        # Method 1: Use the function
        results = load_documents_from_directory(self.test_dir)

        # Verify the results
        assert results is not None
        assert len(results) == 2  # Two test files

        # Method 2: Use the class directly
        loader = DirectoryLoader(directory_path=self.test_dir)
        results2 = loader.load()

        # Verify both methods produce similar results
        assert len(results) == len(results2)

        # Verify the content of the files
        filenames = [doc["metadata"]["filename"] for doc in results]
        assert "test.txt" in filenames
        assert "test.json" in filenames

        # Verify the content of at least one file
        txt_docs = [doc for doc in results if doc["metadata"]["filename"] == "test.txt"]
        assert len(txt_docs) == 1
        assert txt_docs[0]["content"] == "This is a test document."

    def test_backward_compatibility_with_document_processing(self):
        """Test backward compatibility with imports from document_processing."""
        # This test verifies that the old import paths still work via re-exports
        # Import directly from document_processing
        from llm_rag.document_processing import DirectoryLoader as OldDirectoryLoader

        # Import from the new path
        from llm_rag.document_processing.loader_api import DirectoryLoader as NewDirectoryLoader

        # Verify both versions work, even if they might not be the same object
        old_loader = OldDirectoryLoader(directory_path=self.test_dir)
        new_loader = NewDirectoryLoader(directory_path=self.test_dir)

        old_result = old_loader.load()
        new_result = new_loader.load()

        # Verify the results are comparable
        assert len(old_result) == len(new_result)

        # There should be the same types of files
        old_filenames = sorted([doc["metadata"]["filename"] for doc in old_result])
        new_filenames = sorted([doc["metadata"]["filename"] for doc in new_result])
        assert old_filenames == new_filenames
