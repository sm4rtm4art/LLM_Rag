#!/usr/bin/env python3
"""Test script for document retrieval.

This script tests the retrieval capabilities of the RAG system
on the test files in data/documents/test_subset/ by default.
"""

import argparse
import logging
import os
import sys
from io import StringIO
from pathlib import Path

# Try to import optional dependencies
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Add the parent directory to the path so we can import the llm_rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Default test directory
DEFAULT_TEST_DIR = Path('data/documents/test_subset')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test document retrieval with vector search')
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(DEFAULT_TEST_DIR),
        help='Directory containing test documents',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help="Directory to save output files (defaults to data_dir + '_output')",
    )
    parser.add_argument(
        '--extract-images',
        action='store_true',
        help='Extract images from documents',
    )
    parser.add_argument(
        '--extract-tables',
        action='store_true',
        help='Extract tables from documents',
    )
    parser.add_argument(
        '--save-tables',
        action='store_true',
        help='Save extracted tables to CSV files',
    )
    parser.add_argument(
        '--use-enhanced-extraction',
        action='store_true',
        help='Use enhanced PDF extraction (if available)',
    )
    parser.add_argument(
        '--use-structural-image-extraction',
        action='store_true',
        help='Use structural image extraction with PyMuPDF instead of text-based detection',
    )
    parser.add_argument(
        '--environment',
        type=str,
        choices=['development', 'production'],
        default='development',
        help=('Environment mode: development (overwrite DB) or production (check duplicates)'),
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing vector store (default in development mode)',
    )
    return parser.parse_args()


def load_documents(
    data_dir,
    extract_images=False,
    extract_tables=False,
    use_enhanced_extraction=False,
    use_structural_image_extraction=False,
):
    """Load documents from the specified directory.

    Args:
        data_dir: Directory containing the documents
        extract_images: Whether to extract images from PDFs
        extract_tables: Whether to extract tables from PDFs
        use_enhanced_extraction: Whether to use enhanced PDF extraction
        use_structural_image_extraction: Whether to use structural image extraction

    Returns:
        List of documents

    """
    logger.info(f'Loading documents from {data_dir}')

    if use_enhanced_extraction:
        try:
            from scripts.analytics.rag_integration import EnhancedPDFProcessor

            logger.info('Using EnhancedPDFProcessor for document loading')
            processor = EnhancedPDFProcessor(
                save_tables=extract_tables,
                save_images=extract_images,
                use_structural_image_extraction=use_structural_image_extraction,
                verbose=True,
            )

            # Process all PDFs in the directory
            results = {}
            pdf_files = list(Path(data_dir).glob('**/*.pdf'))

            for pdf_file in pdf_files:
                pdf_path = str(pdf_file)
                try:
                    result = processor.process_pdf(pdf_path)
                    if 'documents' in result:
                        results[pdf_path] = result['documents']
                except Exception as e:
                    logger.error(f'Error processing {pdf_path}: {e}')

            # Flatten the list of documents
            documents = []
            for docs in results.values():
                documents.extend(docs)

            return documents

        except ImportError as e:
            logger.error(f'Error importing EnhancedPDFProcessor: {e}')
            logger.warning('Falling back to standard document loading')

    try:
        from llm_rag.document_processing.loaders import DirectoryLoader  # noqa: E402

        # Create output directory
        output_dir = f'{data_dir}_output' if isinstance(data_dir, str) else f'{str(data_dir)}_output'
        os.makedirs(output_dir, exist_ok=True)

        loader = DirectoryLoader(
            directory_path=data_dir,
            recursive=True,
            extract_images=extract_images,
            extract_tables=extract_tables,
            use_enhanced_extraction=use_enhanced_extraction,
            output_dir=output_dir,
        )
        documents = loader.load()
        logger.info(f'Loaded {len(documents)} document chunks')

        # Count document types
        doc_types = {}
        content_types = {}
        for doc in documents:
            metadata = doc.get('metadata', {})
            doc_type = metadata.get('filetype', 'unknown')
            content_type = metadata.get('content_type', 'text')

            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            content_types[content_type] = content_types.get(content_type, 0) + 1

        logger.info(f'Document types: {doc_types}')
        logger.info(f'Content types: {content_types}')

        return documents
    except Exception as e:
        logger.error(f'Error loading documents: {e}')
        return []


def process_documents(documents):
    """Process documents for retrieval testing."""
    logger.info('Processing documents')

    try:
        # Try to use RecursiveTextChunker
        from llm_rag.document_processing.chunking import RecursiveTextChunker  # noqa: E402

        chunker = RecursiveTextChunker(
            chunk_size=1000,
            chunk_overlap=200,
        )
        processed_docs = chunker.split_documents(documents)
        logger.info(f'Split into {len(processed_docs)} chunks')

        # Print sample of processed documents
        print('\n=== Sample of Processed Document Chunks ===')
        for i, doc in enumerate(processed_docs[:3]):  # Show first 3 chunks
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})

            # Get source file name
            source = metadata.get('source', 'unknown')
            source_file = Path(source).name if source else 'unknown'

            # Get content type
            # file_type = metadata.get("filetype", "unknown").upper()  # Unused variable
            content_type = metadata.get('content_type', 'text').upper()

            # Print header with source info
            print(f'\n[{i + 1}] {content_type} - {source_file}')

            # Print metadata
            print(f'Page: {metadata.get("page_num", "N/A")}')
            if 'section_title' in metadata:
                print(f'Section: {metadata.get("section_title", "")}')
            if 'table_index' in metadata:
                print(f'Table: {metadata.get("table_index", "")}')
            if 'image_index' in metadata:
                print(f'Image: {metadata.get("image_index", "")}')

            # Print content (truncate if too long)
            content_preview = content[:200] + '...' if len(content) > 200 else content
            print(f'\nContent: {content_preview}')
            print('-' * 80)

        return processed_docs
    except Exception as e:
        logger.error(f'Error processing documents: {e}')
        return []


def create_vector_store(documents, environment='development', overwrite=None):
    """Create a vector store for testing retrieval."""
    logger.info('Creating vector store')

    try:
        # Prepare documents for vector store
        texts = []
        metadatas = []
        ids = []

        for i, doc in enumerate(documents):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})

            # Skip empty content
            if not content or content.strip() == '':
                logger.warning(f'Skipping empty content from {metadata.get("source", "unknown")}')
                continue

            texts.append(content)
            metadatas.append(metadata)
            ids.append(f'doc_{i}')

        # Create vector store
        from llm_rag.vectorstore.chroma import ChromaVectorStore  # noqa: E402

        # Determine if we should overwrite based on environment if not explicitly set
        if overwrite is None:
            overwrite = environment == 'development'

        # Create vector store with appropriate settings
        persist_directory = 'test_chroma_db'
        vector_store = ChromaVectorStore(
            collection_name='test_retrieval',
            persist_directory=persist_directory,
            overwrite=overwrite,
            environment=environment,
        )

        # Add documents to vector store
        vector_store.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f'Added {len(texts)} documents to vector store')
        return vector_store
    except Exception as e:
        logger.error(f'Error creating vector store: {e}')
        return None


def test_retrieval(vectorstore):
    """Test document retrieval."""
    logger.info('Testing retrieval')

    test_queries = [
        'What is RAG?',
        'How does retrieval augmented generation work?',
        'What are embeddings?',
        'Explain vector databases',
        # Add queries to test table and image retrieval
        'Show me tables with data',
        'Find images in documents',
    ]

    for query in test_queries:
        print(f'\n\n=== Query: {query} ===')
        try:
            # Retrieve documents
            retrieved_docs = vectorstore.similarity_search(query, k=3)
            print(f'Retrieved {len(retrieved_docs)} documents')

            # Print retrieved documents
            for i, doc in enumerate(retrieved_docs):
                # Handle Document objects correctly
                # The Document object might have page_content and metadata
                # attributes instead of being a dict with get method
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                else:
                    # Fallback to dictionary-like access
                    content = doc.get('content', '') if hasattr(doc, 'get') else str(doc)
                    metadata = doc.get('metadata', {}) if hasattr(doc, 'get') else {}

                # Get source file name
                source = metadata.get('source', '') if hasattr(metadata, 'get') else str(metadata)
                source_file = Path(source).name if source else 'unknown'

                # Get content type
                # file_type = metadata.get("filetype", "unknown").upper()  # Unused variable
                content_type = metadata.get('content_type', 'text').upper()

                # Print header with source info
                print(f'\n[{i + 1}] {source_file} - {content_type}')

                # Print additional metadata based on content type
                if content_type == 'table':
                    print(f'Table Index: {metadata.get("table_index", "N/A")}')
                elif content_type == 'image':
                    print(f'Image Index: {metadata.get("image_index", "N/A")}')
                    print(f'Image Path: {metadata.get("image_path", "N/A")}')

                # Print content (truncate if too long)
                content_preview = content[:500] + '...' if len(content) > 500 else content
                print(f'\nContent: {content_preview}')
                print('-' * 80)
        except Exception as e:
            logger.error(f"Error retrieving documents for query '{query}': {e}")


def print_sample_documents(documents, title='Sample of Loaded Documents', num_samples=5):
    """Print a sample of documents for inspection."""
    print(f'\n{title}')
    print('=' * 80)

    # Limit to the specified number of samples
    samples = documents[:num_samples]

    for i, doc in enumerate(samples):
        content = doc.get('content', '')
        metadata = doc.get('metadata', {})

        # Get source file
        source_file = metadata.get('source', 'unknown')

        # Get content type
        content_type = metadata.get('content_type', 'text').upper()

        # Print header with source info
        print(f'\n[{i + 1}] {content_type} - {source_file}')

        # Print metadata
        print(f'Page: {metadata.get("page_num", "N/A")}')
        if 'section_title' in metadata:
            print(f'Section: {metadata.get("section_title", "")}')
        if 'table_index' in metadata:
            print(f'Table: {metadata.get("table_index", "")}')
        if 'image_index' in metadata:
            print(f'Image: {metadata.get("image_index", "")}')

        # Print content (truncate if too long)
        content_preview = content[:200] + '...' if len(content) > 200 else content
        print(f'\nContent: {content_preview}')
        print('-' * 80)


def save_tables_to_csv(documents, output_dir):
    """Save extracted tables as CSV files for easier access.

    Args:
        documents: List of documents containing tables
        output_dir: Directory to save the CSV files

    """
    if not POLARS_AVAILABLE:
        logger.error('polars is required to save tables as CSV. Install with: pip install polars')
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Filter for table documents
    table_documents = [doc for doc in documents if doc.get('metadata', {}).get('content_type') == 'table']

    if not table_documents:
        logger.warning('No tables found to save')
        return

    logger.info(f'Saving {len(table_documents)} tables to {output_dir}')

    # Group tables by source file
    tables_by_source = {}
    for doc in table_documents:
        metadata = doc.get('metadata', {})
        source = metadata.get('source', 'unknown')
        if isinstance(source, Path):
            source = source.name

        if source not in tables_by_source:
            tables_by_source[source] = []

        tables_by_source[source].append(doc)

    # Save tables for each source file
    for source, tables in tables_by_source.items():
        source_dir = os.path.join(output_dir, source.replace('.', '_'))
        os.makedirs(source_dir, exist_ok=True)

        for i, table_doc in enumerate(tables):
            metadata = table_doc.get('metadata', {})
            table_index = metadata.get('table_index', i)
            content = table_doc.get('content', '')

            # Try to convert the table content to a DataFrame
            try:
                # First attempt: Try to parse as CSV
                df = pl.read_csv(StringIO(content), separator=',', ignore_errors=True)

                # Save to CSV
                csv_path = os.path.join(source_dir, f'table_{table_index}.csv')
                df.write_csv(csv_path)
                logger.info(f'Saved table {table_index} from {source} to {csv_path}')

                # Also save as JSON for better structure preservation
                try:
                    json_path = os.path.join(source_dir, f'table_{table_index}.json')
                    df.write_json(json_path)
                except Exception as e:
                    logger.warning(f'Could not save as JSON: {e}')

            except Exception as e:
                logger.warning(f'Could not parse table {table_index} from {source} as CSV: {e}')

                # Fallback: Save raw content
                raw_path = os.path.join(source_dir, f'table_{table_index}_raw.txt')
                with open(raw_path, 'w') as f:
                    f.write(content)
                logger.info(f'Saved raw table content to {raw_path}')

    logger.info(f'Finished saving tables to {output_dir}')


def main():
    """Parse arguments and run the retrieval test."""
    args = parse_arguments()

    # Set up data and output directories
    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f'{data_dir}_output')

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load documents
    documents = load_documents(
        data_dir,
        extract_images=args.extract_images,
        extract_tables=args.extract_tables,
        use_enhanced_extraction=args.use_enhanced_extraction,
        use_structural_image_extraction=args.use_structural_image_extraction,
    )

    if not documents:
        logger.error('No documents loaded. Exiting.')
        return

    logger.info(f'Loaded {len(documents)} documents')

    # Process documents
    processed_docs = process_documents(documents)
    if not processed_docs:
        logger.error('No documents processed. Exiting.')
        return

    logger.info(f'Processed {len(processed_docs)} documents')

    # Print sample documents
    print_sample_documents(processed_docs)

    # Save tables to CSV if requested
    if args.save_tables and args.extract_tables:
        save_tables_to_csv(processed_docs, output_dir)

    # Create vector store
    vectorstore = create_vector_store(processed_docs, environment=args.environment, overwrite=args.overwrite)
    if not vectorstore:
        logger.error('Failed to create vector store. Exiting.')
        return

    # Test retrieval
    test_retrieval(vectorstore)


if __name__ == '__main__':
    main()
