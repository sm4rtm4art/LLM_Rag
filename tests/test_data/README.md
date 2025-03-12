# LLM RAG Examples

This directory contains example scripts demonstrating how to use the LLM RAG framework with various document types.

## DIN XML Document Examples

The `din_document.xml` file is a DIN-formatted XML document that follows a standardized structure for technical documentation. It's used in several example scripts to demonstrate how the XMLLoader can be used to process structured XML content for RAG systems.

### Running the Examples

First, make sure you have installed the required dependencies:

```bash
# For basic functionality
pip install -e .

# For advanced RAG demo with embeddings
pip install sentence-transformers
```

Then, run the examples from the project root directory:

```bash
# Basic loading examples
python examples/din_loader_demo.py

# Advanced RAG demo (requires sentence-transformers)
python examples/din_rag_demo.py

# XML namespace utilities
python examples/xml_namespace_utils.py
```

### Example Scripts

1. **din_loader_demo.py**

   - Demonstrates various ways to load and extract information from the DIN document
   - Shows how to extract metadata, split into sections, extract definitions, code examples, and tables

2. **din_rag_demo.py**

   - Creates a simple RAG system using the DIN document
   - Implements vector-based search with sentence-transformers
   - Shows how to answer questions using retrieved content

3. **xml_namespace_utils.py**
   - Helper functions for working with namespaced XML documents
   - Examples of extracting namespaces, finding elements by tag, and handling definitions

### Further Documentation

For more detailed information about the DIN document structure and how to use it with the XMLLoader, see the `README_din_document.md` file in this directory.

## What You Can Now Do

### 1. Load the DIN Document in Different Ways

You can use the XMLLoader to load the document as a whole or in various structured ways:

```python
from llm_rag.document_processing.loaders import XMLLoader

# Load the entire document
loader = XMLLoader("examples/din_document.xml")
documents = loader.load()

# Load by sections
loader = XMLLoader(
    "examples/din_document.xml",
    split_by_tag="din:section",
    metadata_tags=["din:title"]
)
sections = loader.load()

# Extract just the definitions
loader = XMLLoader(
    "examples/din_document.xml",
    split_by_tag="din:definition",
    content_tags=["din:term", "din:description"]
)
definitions = loader.load()
```

### 2. Use It in a RAG System

The advanced demo (`din_rag_demo.py`) shows how to use the DIN document in a full RAG system:

- It loads each section as a separate document
-
