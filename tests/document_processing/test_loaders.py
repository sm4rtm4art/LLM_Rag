"""Tests for document loaders.

This module tests both the individual loaders and the modular structure to ensure
backward compatibility is maintained.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from llm_rag.document_processing.loaders import (
    CSVLoader,
    DirectoryLoader,
    EnhancedPDFLoader,
    JSONLoader,
    PDFLoader,
    TextFileLoader,
    WebLoader,
    WebPageLoader,
    XMLLoader,
)


class TestLoadersImports(unittest.TestCase):
    """Test that all loaders can be imported correctly."""

    def test_imports(self):
        """Test that all loaders can be imported."""
        # This test passes if the imports above work
        self.assertTrue(TextFileLoader)
        self.assertTrue(PDFLoader)
        self.assertTrue(EnhancedPDFLoader)
        self.assertTrue(CSVLoader)
        self.assertTrue(JSONLoader)
        self.assertTrue(XMLLoader)
        self.assertTrue(WebLoader)
        self.assertTrue(WebPageLoader)
        self.assertTrue(DirectoryLoader)


class TestTextFileLoader(unittest.TestCase):
    """Test the TextFileLoader."""

    def setUp(self):
        """Set up a temporary file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = Path(self.temp_dir.name) / "test.txt"
        with open(self.test_file, "w") as f:
            f.write("This is a test file.\nIt has multiple lines.\n")

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_load(self):
        """Test loading from a text file."""
        loader = TextFileLoader(self.test_file)
        docs = loader.load()

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content"], "This is a test file.\nIt has multiple lines.\n")
        self.assertEqual(docs[0]["metadata"]["source"], str(self.test_file))
        self.assertEqual(docs[0]["metadata"]["filename"], "test.txt")
        self.assertEqual(docs[0]["metadata"]["filetype"], "text")


class TestJSONLoader(unittest.TestCase):
    """Test the JSONLoader."""

    def setUp(self):
        """Set up a temporary file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = Path(self.temp_dir.name) / "test.json"

        test_data = {
            "title": "Test Document",
            "content": "This is the content of the test document.",
            "metadata": {"author": "Test Author", "date": "2023-04-15"},
            "items": [{"name": "Item 1", "value": 10}, {"name": "Item 2", "value": 20}],
        }

        with open(self.test_file, "w") as f:
            json.dump(test_data, f)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_load_basic(self):
        """Test basic loading from a JSON file."""
        loader = JSONLoader(self.test_file)
        docs = loader.load()

        self.assertEqual(len(docs), 1)
        self.assertIn("title", docs[0]["content"])
        self.assertIn("content", docs[0]["content"])
        self.assertEqual(docs[0]["metadata"]["source"], str(self.test_file))
        self.assertEqual(docs[0]["metadata"]["filename"], "test.json")
        self.assertEqual(docs[0]["metadata"]["filetype"], "json")

    def test_load_with_content_key(self):
        """Test loading with a specific content key."""
        loader = JSONLoader(self.test_file, content_key="content")
        docs = loader.load()

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["content"], "This is the content of the test document.")

    def test_load_with_metadata_keys(self):
        """Test loading with specific metadata keys."""
        loader = JSONLoader(self.test_file, content_key="content", metadata_keys=["title"])
        docs = loader.load()

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["metadata"]["title"], "Test Document")

    def test_load_with_jq_schema(self):
        """Test loading with a JQ-like schema."""
        loader = JSONLoader(self.test_file, jq_schema="items")
        docs = loader.load()

        self.assertEqual(len(docs), 2)
        self.assertIn("Item 1", docs[0]["content"])
        self.assertIn("Item 2", docs[1]["content"])


class TestCSVLoader(unittest.TestCase):
    """Test the CSVLoader."""

    def setUp(self):
        """Set up a temporary file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = Path(self.temp_dir.name) / "test.csv"

        with open(self.test_file, "w") as f:
            f.write("id,name,description\n")
            f.write("1,Item 1,This is item 1\n")
            f.write("2,Item 2,This is item 2\n")
            f.write("3,Item 3,This is item 3\n")

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_load(self):
        """Test loading from a CSV file."""
        loader = CSVLoader(self.test_file)
        docs = loader.load()

        self.assertEqual(len(docs), 3)
        self.assertIn("Item 1", docs[0]["content"])
        self.assertIn("Item 2", docs[1]["content"])
        self.assertIn("Item 3", docs[2]["content"])

    def test_load_with_content_columns(self):
        """Test loading with specific content columns."""
        loader = CSVLoader(self.test_file, content_columns=["description"])
        docs = loader.load()

        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0]["content"], "This is item 1")
        self.assertEqual(docs[1]["content"], "This is item 2")
        self.assertEqual(docs[2]["content"], "This is item 3")

    def test_load_with_metadata_columns(self):
        """Test loading with specific metadata columns."""
        loader = CSVLoader(self.test_file, content_columns=["description"], metadata_columns=["id", "name"])
        docs = loader.load()

        self.assertEqual(len(docs), 3)
        self.assertTrue(docs[0]["metadata"]["id"] == "1" or docs[0]["metadata"]["id"] == 1)
        self.assertEqual(docs[0]["metadata"]["name"], "Item 1")


class TestXMLLoader(unittest.TestCase):
    """Test the XMLLoader."""

    def setUp(self):
        """Set up a temporary file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file = Path(self.temp_dir.name) / "test.xml"

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <library>
            <book id="1">
                <title>The Great Gatsby</title>
                <author>F. Scott Fitzgerald</author>
                <description>A novel set in the Jazz Age...</description>
            </book>
            <book id="2">
                <title>To Kill a Mockingbird</title>
                <author>Harper Lee</author>
                <description>The story of the Finch family...</description>
            </book>
        </library>
        """

        with open(self.test_file, "w") as f:
            f.write(xml_content)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_load(self):
        """Test loading from an XML file."""
        loader = XMLLoader(self.test_file)
        docs = loader.load()

        self.assertEqual(len(docs), 1)
        self.assertIn("The Great Gatsby", docs[0]["content"])
        self.assertIn("To Kill a Mockingbird", docs[0]["content"])

    def test_load_with_split_by_tag(self):
        """Test loading with splitting by tag."""
        loader = XMLLoader(self.test_file, split_by_tag="book")
        docs = loader.load()

        self.assertEqual(len(docs), 2)
        self.assertIn("The Great Gatsby", docs[0]["content"])
        self.assertIn("To Kill a Mockingbird", docs[1]["content"])

    def test_load_with_content_tags(self):
        """Test loading with specific content tags."""
        loader = XMLLoader(self.test_file, content_tags=["title", "author"])
        docs = loader.load()

        self.assertEqual(len(docs), 1)
        self.assertIn("title: The Great Gatsby", docs[0]["content"])
        self.assertIn("author: F. Scott Fitzgerald", docs[0]["content"])


class TestWebLoader(unittest.TestCase):
    """Test the WebLoader with mocked responses."""

    def test_init(self):
        """Test initialization of WebLoader."""
        loader = WebLoader("https://example.com")
        self.assertEqual(loader.web_path, "https://example.com")


class TestDirectoryLoader(unittest.TestCase):
    """Test the DirectoryLoader."""

    def setUp(self):
        """Set up a temporary directory with test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create a text file
        self.text_file = self.test_dir / "test.txt"
        with open(self.text_file, "w") as f:
            f.write("This is a test text file.\n")

        # Create a JSON file
        self.json_file = self.test_dir / "test.json"
        with open(self.json_file, "w") as f:
            json.dump({"title": "Test JSON", "content": "This is a test JSON file."}, f)

        # Create a CSV file
        self.csv_file = self.test_dir / "test.csv"
        with open(self.csv_file, "w") as f:
            f.write("id,name\n")
            f.write("1,Test CSV\n")

        # Create a subdirectory
        self.sub_dir = self.test_dir / "subdir"
        os.makedirs(self.sub_dir, exist_ok=True)

        # Create a file in the subdirectory
        self.sub_file = self.sub_dir / "subtest.txt"
        with open(self.sub_file, "w") as f:
            f.write("This is a test file in a subdirectory.\n")

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_load(self):
        """Test loading from a directory."""
        loader = DirectoryLoader(self.test_dir)
        docs = loader.load()

        self.assertGreaterEqual(len(docs), 3)  # At least the 3 files in the main directory

    def test_load_non_recursive(self):
        """Test loading from a directory non-recursively."""
        loader = DirectoryLoader(self.test_dir, recursive=False)
        docs = loader.load()

        # Should not include files from subdirectories
        for doc in docs:
            self.assertNotIn("subdir", doc["metadata"]["source"])

    def test_load_with_glob_pattern(self):
        """Test loading with a specific glob pattern."""
        loader = DirectoryLoader(self.test_dir, glob_pattern="*.txt")
        docs = loader.load()

        # Should only include .txt files
        for doc in docs:
            self.assertTrue(doc["metadata"]["source"].endswith(".txt"))


if __name__ == "__main__":
    unittest.main()
