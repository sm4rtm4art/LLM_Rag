#!/usr/bin/env python
"""Check the content of PDF files directly using PyPDF."""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        str: Extracted text from the PDF file, or None if extraction fails

    """
    try:
        import pypdf

        with open(pdf_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)

            # Extract text from each page
            text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + '\n\n'

            return text
    except ImportError:
        logger.error("PyPDF library not found. Please install it with 'pip install pypdf'")
        return None
    except Exception as e:
        logger.error(f'Error extracting text from {pdf_path}: {e}')
        return None


def check_pdf_content(pdf_dir='data/documents/test_subset'):
    """Check the content of PDF files.

    Args:
        pdf_dir: Directory containing PDF files

    Returns:
        None

    """
    # Convert to Path object if it's a string
    pdf_dir = Path(pdf_dir)

    # Find all PDF files
    pdf_files = list(pdf_dir.glob('**/*.pdf'))
    logger.info(f'Found {len(pdf_files)} PDF files')

    # Extract text from each PDF file
    for pdf_file in pdf_files:
        logger.info(f'Processing {pdf_file}')
        text = extract_text_from_pdf(pdf_file)

        if text:
            # Print the first 500 characters of the text
            logger.info(f'Content sample: {text[:500]}...')
            logger.info('-' * 50)
        else:
            logger.warning(f'Failed to extract text from {pdf_file}')
            logger.info('-' * 50)


def main():
    """Run the PDF content check."""
    check_pdf_content()


if __name__ == '__main__':
    main()
