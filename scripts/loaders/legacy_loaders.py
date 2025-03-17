#!/usr/bin/env python3
"""Test script for legacy loaders.py.

This script tests that the legacy loaders from loaders_old.py still work after refactoring.
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Get the project root directory (2 levels up from the script location)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent

# Add the project root to the Python path
sys.path.insert(0, str(project_root))

# Import the legacy loaders
from src.llm_rag.document_processing import loaders_old

# Get the loader classes from the module
TextFileLoader = loaders_old.TextFileLoader
CSVLoader = loaders_old.CSVLoader
JSONLoader = loaders_old.JSONLoader
DirectoryLoader = loaders_old.DirectoryLoader
WebPageLoader = loaders_old.WebPageLoader


def test_text_loader():
    """Test loading a text file with the legacy TextFileLoader."""
    # Create a test text file
    test_file = Path("test_file.txt")
    with open(test_file, "w") as f:
        f.write("This is a test document.\nIt has multiple lines.\nThis is for testing the TextFileLoader.")

    try:
        # Load the text file
        loader = TextFileLoader(test_file)
        documents = loader.load()

        # Verify the results
        logger.info(f"Loaded {len(documents)} documents from text file")
        logger.info(f"Content: {documents[0]['content'][:50]}...")
        logger.info(f"Metadata: {documents[0]['metadata']}")

        return len(documents) > 0
    except Exception as e:
        logger.error(f"Error testing TextFileLoader: {e}")
        return False
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


def test_csv_loader():
    """Test loading a CSV file with the legacy CSVLoader."""
    # Create a test CSV file
    test_file = Path("test_file.csv")
    with open(test_file, "w") as f:
        f.write("id,title,content\n")
        f.write("1,Test Document 1,This is the content of document 1\n")
        f.write("2,Test Document 2,This is the content of document 2\n")

    try:
        # Load the CSV file
        loader = CSVLoader(test_file, content_columns=["content"], metadata_columns=["id", "title"])
        documents = loader.load()

        # Verify the results
        logger.info(f"Loaded {len(documents)} documents from CSV file")
        for doc in documents:
            logger.info(f"Content: {doc['content'][:30]}...")
            logger.info(f"Metadata: {doc['metadata']}")

        return len(documents) == 2
    except Exception as e:
        logger.error(f"Error testing CSVLoader: {e}")
        return False
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


def test_json_loader():
    """Test loading a JSON file with the legacy JSONLoader."""
    # Create a test JSON file
    test_file = Path("test_file.json")
    with open(test_file, "w") as f:
        f.write("""
        {
            "documents": [
                {
                    "id": 1,
                    "title": "Test Document 1",
                    "content": "This is the content of document 1"
                },
                {
                    "id": 2,
                    "title": "Test Document 2",
                    "content": "This is the content of document 2"
                }
            ]
        }
        """)

    try:
        # Load the JSON file - note: legacy JSONLoader doesn't support jq_schema in the same way
        # so we just use content_key to extract content
        loader = JSONLoader(test_file, content_key="content")
        documents = loader.load()

        # Verify the results
        logger.info(f"Loaded {len(documents)} documents from JSON file")
        for doc in documents:
            logger.info(f"Content: {doc['content'][:30]}...")
            logger.info(f"Metadata: {doc['metadata']}")

        # The test passes if we loaded at least one document
        return len(documents) > 0
    except Exception as e:
        logger.error(f"Error testing JSONLoader: {e}")
        return False
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


def test_directory_loader():
    """Test loading files from a directory with the legacy DirectoryLoader."""
    # Create a test directory with files
    test_dir = Path("test_dir")
    test_dir.mkdir(exist_ok=True)

    # Create a few text files
    for i in range(3):
        file_path = test_dir / f"test_file_{i}.txt"
        with open(file_path, "w") as f:
            f.write(f"This is test file {i}.\nIt contains sample content for testing the DirectoryLoader.")

    try:
        # Load files from the directory
        loader = DirectoryLoader(test_dir, glob_pattern="*.txt")
        documents = loader.load()

        # Verify the results
        logger.info(f"Loaded {len(documents)} documents from directory")
        for i, doc in enumerate(documents):
            logger.info(f"Document {i} content: {doc['content'][:30]}...")
            logger.info(f"Document {i} metadata: {doc['metadata']}")

        return len(documents) == 3
    except Exception as e:
        logger.error(f"Error testing DirectoryLoader: {e}")
        return False
    finally:
        # Clean up
        if test_dir.exists():
            for file in test_dir.glob("*.txt"):
                file.unlink()
            test_dir.rmdir()


def test_web_loader():
    """Test the WebPageLoader with a mock to avoid actual web requests."""
    from unittest.mock import MagicMock, patch

    try:
        # Mock the requests.get method
        with patch("requests.get") as mock_get:
            # Set up the mock response
            mock_response = MagicMock()
            mock_response.text = "<html><body><h1>Test Page</h1><p>This is a test webpage.</p></body></html>"
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            # Use the WebPageLoader
            loader = WebPageLoader("https://example.com")
            documents = loader.load()

            # Verify the results
            logger.info(f"Loaded {len(documents)} documents from web page")
            logger.info(f"Content: {documents[0]['content'][:50]}...")
            logger.info(f"Metadata: {documents[0]['metadata']}")

            return len(documents) > 0
    except Exception as e:
        logger.error(f"Error testing WebPageLoader: {e}")
        return False


def main():
    """Run all the loader tests."""
    logger.info("Testing legacy loaders...")

    tests = {
        "TextFileLoader": test_text_loader,
        "CSVLoader": test_csv_loader,
        "JSONLoader": test_json_loader,
        "DirectoryLoader": test_directory_loader,
        "WebPageLoader": test_web_loader,
    }

    results = {}
    for name, test_func in tests.items():
        logger.info(f"\nTesting {name}...")
        results[name] = test_func()
        status = "PASSED" if results[name] else "FAILED"
        logger.info(f"{name} test {status}")

    # Print summary
    logger.info("\nTest Summary:")
    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{name}: {status}")

    all_passed = all(results.values())
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    main()
