#!/usr/bin/env python3
"""Test script for HTML parsing functionality.

This script demonstrates the different features of the HTML parser:
- Basic text extraction
- Metadata extraction
- HTML mode
- Markdown mode
- Image extraction
"""

import logging

from llm_rag.document_processing.loaders import WebLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_html_parsing(url: str = 'https://example.com') -> None:
    """Test HTML parsing with different modes and features.

    Parameters
    ----------
    url : str, optional
        URL to test with, by default "https://example.com"

    """
    logger.info(f'Testing HTML parsing with URL: {url}')

    # Test 1: Basic text extraction with metadata
    logger.info('\nTest 1: Basic text extraction with metadata')
    loader = WebLoader(url=url, extract_metadata=True)
    docs = loader.load()
    metadata = docs[0]['metadata']
    logger.info(f'Title: {metadata.get("title", "N/A")}')
    logger.info(f'Description: {metadata.get("description", "N/A")}')
    logger.info(f'Author: {metadata.get("author", "N/A")}')
    logger.info(f'Content preview: {docs[0]["content"][:200]}...')

    # Test 2: HTML mode (raw HTML)
    logger.info('\nTest 2: HTML mode (raw HTML)')
    loader = WebLoader(url=url, html_mode='html')
    docs = loader.load()
    logger.info(f'Raw HTML preview: {docs[0]["content"][:200]}...')

    # Test 3: Markdown mode (if html2text is available)
    logger.info('\nTest 3: Markdown mode')
    loader = WebLoader(url=url, html_mode='markdown')
    docs = loader.load()
    logger.info(f'Markdown preview: {docs[0]["content"][:200]}...')

    # Test 4: Image extraction
    logger.info('\nTest 4: Image extraction')
    loader = WebLoader(url=url, extract_images=True)
    docs = loader.load()
    images = docs[0]['metadata'].get('image_urls', [])
    logger.info(f'Found {len(images)} images')
    for img in images[:3]:  # Show first 3 images
        logger.info(f'Image URL: {img}')


def main() -> None:
    """Run the HTML parser tests."""
    # Test with a real website
    test_url = 'https://www.python.org'
    logger.info('Starting HTML parser tests...')
    test_html_parsing(test_url)
    logger.info('HTML parser tests completed.')


if __name__ == '__main__':
    main()
