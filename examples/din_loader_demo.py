#!/usr/bin/env python
"""Demo script for using XMLLoader with a DIN-formatted XML document.

Shows different loading strategies including section-by-section loading,
metadata extraction, and term definition extraction.
"""

import os

from llm_rag.document_processing.loaders import XMLLoader


def print_document(doc, full_content=False):
    """Pretty print a document."""
    print('-' * 80)
    print('Document Metadata:')
    for key, value in doc['metadata'].items():
        print(f'  {key}: {value}')
    print('-' * 80)

    if full_content:
        print('Content:')
        print(doc['content'])
    else:
        # Print just the first few lines
        lines = doc['content'].split('\n')
        preview = '\n'.join(lines[:5])
        print('Content Preview (first 5 lines):')
        print(preview)
        print(f'...and {len(lines) - 5} more lines')
    print('-' * 80)
    print()


def basic_loading():
    """Load the entire DIN document."""
    print('\n=== BASIC LOADING ===\n')

    loader = XMLLoader('examples/din_document.xml')
    documents = loader.load()

    print(f'Loaded {len(documents)} document(s)')
    print_document(documents[0])


def load_with_metadata():
    """Extract metadata from specific tags during loading."""
    print('\n=== LOADING WITH METADATA EXTRACTION ===\n')

    loader = XMLLoader(
        'examples/din_document.xml',
        metadata_tags=[
            'din:identifier',
            'din:title',
            'din:version',
            'din:language',
            'din:publicationDate',
            'din:lastModified',
        ],
    )
    documents = loader.load()

    print(f'Loaded {len(documents)} document(s) with specific metadata')
    print_document(documents[0])


def load_section_by_section():
    """Split the document into sections and load each as a separate document.

    This creates individual documents for better semantic chunking.
    """
    print('\n=== LOADING SECTION BY SECTION ===\n')

    loader = XMLLoader('examples/din_document.xml', split_by_tag='din:section', metadata_tags=['din:title'])
    documents = loader.load()

    print(f'Loaded {len(documents)} section documents')

    # Display first few sections
    for i, doc in enumerate(documents[:3]):
        print(f'Section {i + 1}:')
        print_document(doc)

    if len(documents) > 3:
        print(f'...and {len(documents) - 3} more sections')


def extract_definitions():
    """Extract term definitions from the document."""
    print('\n=== EXTRACTING DEFINITIONS ===\n')

    loader = XMLLoader(
        'examples/din_document.xml', split_by_tag='din:definition', content_tags=['din:term', 'din:description']
    )
    documents = loader.load()

    print(f'Extracted {len(documents)} definitions')

    # Display all definitions
    for i, doc in enumerate(documents):
        print(f'Definition {i + 1}:')
        print_document(doc, full_content=True)


def extract_code_examples():
    """Extract code examples from the document."""
    print('\n=== EXTRACTING CODE EXAMPLES ===\n')

    loader = XMLLoader('examples/din_document.xml', content_tags=['din:code'], include_tags_in_text=False)
    documents = loader.load()

    print('Extracted code examples from the document')
    print_document(documents[0], full_content=True)


def extract_tables():
    """Extract tables from the document."""
    print('\n=== EXTRACTING TABLES ===\n')

    loader = XMLLoader(
        'examples/din_document.xml',
        split_by_tag='din:table',
        metadata_tags=['din:caption'],
        content_tags=['din:row', 'din:cell', 'din:header'],
    )
    documents = loader.load()

    print(f'Extracted {len(documents)} tables')

    for i, doc in enumerate(documents):
        print(f'Table {i + 1}:')
        print_document(doc, full_content=True)


if __name__ == '__main__':
    # Make sure we can find the example file
    if not os.path.exists('examples/din_document.xml'):
        print('Error: examples/din_document.xml not found')
        print('Run this script from the project root directory')
        exit(1)

    # Run all the demo functions
    basic_loading()
    load_with_metadata()
    load_section_by_section()
    extract_definitions()
    extract_code_examples()
    extract_tables()
