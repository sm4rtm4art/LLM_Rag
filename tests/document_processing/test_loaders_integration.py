"""Integration tests for document loaders.

This module tests the loaders' compatibility with the rest of the system,
including backward compatibility with older code.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

# For backward compatibility, import directly from document_processing
from llm_rag.document_processing import DirectoryLoader
from llm_rag.document_processing import TextFileLoader as TopLevelTextFileLoader

# Test import paths
from llm_rag.document_processing.loaders import TextFileLoader as NewTextFileLoader


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility between old and new loaders."""

    def setUp(self):
        """Set up a temporary file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = Path(self.temp_dir.name) / "test.txt"
        with open(self.test_file, "w") as f:
            f.write("This is a test file for compatibility testing.\n")

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_imports_work(self):
        """Test that both old and new import paths work."""
        self.assertTrue(NewTextFileLoader)
        self.assertTrue(TopLevelTextFileLoader)

    def test_new_loader(self):
        """Test that the new loader works."""
        loader = NewTextFileLoader(self.test_file)
        docs = loader.load()
        self.assertEqual(len(docs), 1)
        self.assertIn("compatibility testing", docs[0]["content"])

    def test_top_level_loader(self):
        """Test that the top-level loader works (backward compatibility)."""
        loader = TopLevelTextFileLoader(self.test_file)
        docs = loader.load()
        self.assertEqual(len(docs), 1)
        self.assertIn("compatibility testing", docs[0]["content"])


class TestPipelineIntegration(unittest.TestCase):
    """Test integration with the RAG pipeline."""

    def setUp(self):
        """Set up test files and data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "data"
        os.makedirs(self.data_dir, exist_ok=True)

        # Create sample files
        self.create_sample_files()

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def create_sample_files(self):
        """Create sample files for testing."""
        # Text file
        with open(self.data_dir / "doc1.txt", "w") as f:
            f.write("This is document 1. It contains information about topic A.")

        # JSON file
        with open(self.data_dir / "doc2.json", "w") as f:
            json.dump(
                {
                    "title": "Document 2",
                    "content": "This document contains information about topic B.",
                    "metadata": {"author": "Test Author"},
                },
                f,
            )

        # CSV file
        with open(self.data_dir / "doc3.csv", "w") as f:
            f.write("id,title,content\n")
            f.write("1,Document 3,This document contains information about topic C.\n")

    def test_directory_loader(self):
        """Test loading multiple document types with DirectoryLoader."""
        loader = DirectoryLoader(self.data_dir)
        docs = loader.load()

        # Check that all documents were loaded
        self.assertGreaterEqual(len(docs), 3)

        # Check that content from each file type is present
        contents = [doc["content"] for doc in docs]
        all_content = "\n".join(contents)

        self.assertIn("topic A", all_content)
        self.assertIn("topic B", all_content)
        self.assertIn("topic C", all_content)

    def test_document_chain(self):
        """Test a chain of document processing operations."""
        # 1. Load documents
        loader = DirectoryLoader(self.data_dir)
        docs = loader.load()

        # 2. Filter documents (simulating a pipeline operation)
        filtered_docs = [doc for doc in docs if "topic B" in doc["content"]]

        # 3. Transform documents (simulating another pipeline operation)
        transformed_docs = []
        for doc in filtered_docs:
            new_doc = doc.copy()
            new_doc["content"] = doc["content"].upper()
            transformed_docs.append(new_doc)

        # Verify results
        self.assertGreaterEqual(len(filtered_docs), 1)
        for doc in transformed_docs:
            self.assertIn("TOPIC B", doc["content"])


if __name__ == "__main__":
    unittest.main()
