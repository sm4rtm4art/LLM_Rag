# Document Processing Module

This module provides a comprehensive solution for processing documents from various sources and formats for use in RAG (Retrieval-Augmented Generation) applications.

## Architecture

The document processing module follows a modular architecture with clear separation of concerns:

```
document_processing/
├── loaders/              # Document loaders for different sources
│   ├── base.py           # Base classes and interfaces
│   ├── factory.py        # Factory functions for creating loaders
│   ├── file_loaders.py   # Loaders for text and CSV files
│   ├── pdf_loaders.py    # Loaders for PDF files
│   ├── json_loader.py    # Loader for JSON files
│   ├── web_loader.py     # Loader for web content
│   ├── directory_loader.py # Loader for directories
│   └── __init__.py       # Public API exports
├── processors/           # Document processors
├── splitters/            # Document splitting strategies
└── loaders.py            # Legacy loader implementations
```

## Key Components

### Document Loaders

Document loaders extract text and metadata from various sources:

1. **Base Interfaces and Registry**:

   - `DocumentLoader`: Abstract base class for all loaders
   - `FileLoader`: Protocol for file-based loaders
   - `DirectoryLoader`: Protocol for directory-based loaders
   - `WebLoader`: Protocol for web-based loaders
   - `LoaderRegistry`: Registry for loader classes based on file extensions

2. **File Loaders**:

   - `TextFileLoader`: Load documents from text files
   - `CSVLoader`: Load documents from CSV files
   - `PDFLoader`: Load documents from PDF files
   - `EnhancedPDFLoader`: Load documents from PDF files with advanced features
   - `JSONLoader`: Load documents from JSON files or JSON Lines

3. **Web Loader**:

   - `WebLoader`: Load documents from web URLs with support for HTML extraction

4. **Directory Loader**:

   - `DirectoryLoader`: Load documents from all files in a directory

5. **Factory Functions**:
   - `load_document`: Load a document from a file using the appropriate loader
   - `load_documents_from_directory`: Load all documents from a directory

### Document Processors

Document processors transform raw documents:

1. **Base Processor**:

   - `DocumentProcessor`: Base class for all processors

2. **Processor Implementations**:
   - Various processors for text normalization, metadata extraction, etc.

## Usage Examples

### Loading Documents

```python
from llm_rag.document_processing.loaders import (
    load_document,
    load_documents_from_directory,
    TextFileLoader,
    CSVLoader,
    JSONLoader,
    PDFLoader,
    WebLoader,
    DirectoryLoader
)

# Load a single document using the factory function
documents = load_document("path/to/document.txt")

# Load from a specific file type
pdf_loader = PDFLoader("path/to/document.pdf")
pdf_documents = pdf_loader.load()

# Load JSON data with specific content and metadata keys
json_loader = JSONLoader(
    "path/to/data.json",
    content_key="text",
    metadata_keys=["title", "author"]
)
json_documents = json_loader.load()

# Load from a web URL
web_loader = WebLoader(
    "https://example.com",
    extract_metadata=True
)
web_documents = web_loader.load()

# Load all documents in a directory
dir_loader = DirectoryLoader(
    "path/to/docs",
    glob_pattern="**/*.pdf",
    recursive=True
)
directory_documents = dir_loader.load()

# Or use the factory function for directories
all_documents = load_documents_from_directory("path/to/docs")
```

### Processing Documents

```python
from llm_rag.document_processing.processors import (
    NormalizationProcessor,
    MetadataProcessor
)

# Create a processor pipeline
processor = NormalizationProcessor() | MetadataProcessor()

# Process documents
processed_documents = processor.process(documents)
```

## Backward Compatibility

The module maintains backward compatibility with the legacy loader implementations. You can continue using the original loader classes, but it's recommended to migrate to the new modular architecture for better maintainability and extensibility.

```python
# Legacy API (still supported)
from llm_rag.document_processing.loaders import LegacyPDFLoader

# New modular API (recommended)
from llm_rag.document_processing.loaders import PDFLoader
```

## Extension

The modular architecture makes it easy to add new loader types:

1. Create a new loader class that implements `DocumentLoader` and appropriate protocol
2. Register it with the registry
3. Update the factory function if needed

Example:

```python
from llm_rag.document_processing.loaders.base import DocumentLoader, FileLoader, registry

class MyCustomLoader(DocumentLoader, FileLoader):
    # Implementation...
    pass

# Register with specific extensions
registry.register(MyCustomLoader, extensions=["custom", "mycustom"])
```

## Future Enhancements

Planned enhancements include:

1. Additional loader types for more file formats
2. Advanced metadata extraction
3. Improved error handling and recovery mechanisms
4. Parallel loading for large document collections
